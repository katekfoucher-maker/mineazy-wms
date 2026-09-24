"""Configuration - environment driven (``.env`` supported)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# project root = parent of the ``wms`` package dir
_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Mineazy"
    debug: bool = False

    # Web session signing key. CHANGE in production via env var.
    secret_key: str = "dev-secret-change-me-to-a-long-random-string"
    session_max_age: int = 60 * 60 * 12   # 12h

    # "Sign up with Google" (see /signup, wms/services/google_oauth.py). From
    # a Google Cloud Console OAuth 2.0 "Web application" client - add every
    # deployment's "<origin>/auth/google/callback" to that client's Authorized
    # redirect URIs. Blank = the signup page shows a "not configured yet"
    # message instead of a broken button. Google requires the redirect URI to
    # be https:// for anything other than localhost, so an http-only
    # deployment (e.g. a bare EC2 IP) can't use this until it has HTTPS.
    google_client_id: str = ""
    google_client_secret: str = ""

    # Database. MySQL is the target backend; SQLite is the zero-setup fallback.
    #   MySQL:  mysql+pymysql://wms:wms_password@localhost:3306/mineazy_wms
    database_url: str = "sqlite:///./wms.db"
    # Path to a CA certificate file, for a MySQL host that requires TLS (e.g.
    # Aiven's free tier). Leave blank for a plain/local connection.
    db_ssl_ca: str = ""

    # Analytics / forecasting defaults
    sales_history_days: int = 90
    service_level: float = 0.95          # z ~ 1.645
    lead_time_days: int = 7
    review_period_days: int = 7
    # Days in transit from dispatch until the branch receives the stock. The
    # weekly allocation tops branches up to cover the review week PLUS this, so
    # they never run dry while a transfer is on the road.
    dispatch_transit_days: int = 3
    # Temporary: stock-on-hand data generally isn't trusted right now, so
    # allocation runs on sales/demand alone - every branch's on-hand reads as
    # 0 for allocation math (see stock.levels_df_for_allocation), meaning
    # "target" is sent in full regardless of what's already on the shelf.
    # Flip back to True once stock-on-hand is trustworthy again; the
    # Inventory page's own numbers are unaffected either way.
    allocation_use_inventory: bool = False
    # Restocking bias. It is safer to over-forecast than to run a branch dry, so
    # the forecast is calibrated to this PRE-FLOOR margin versus the reference
    # sales level; the min-units floor below then lifts the served total further
    # over. Negative = sit under before the floor (the floor does the lifting).
    weekly_safety_margin: float = 0.05
    # A bulk-order week above the robust cap counts this much of its excess
    # toward the mean (0.75 = three quarters), but never more than
    # weekly_peak_ceil x the product's typical (median non-zero) sale, so a peak
    # is taken seriously without one outlier running away with the level. The
    # forecast is then clamped ASYMMETRICALLY to that spike-damped mean: never
    # below (1 - down_band), free up to (1 + up_band).
    weekly_spike_influence: float = 0.75
    weekly_peak_ceil: float = 3.0           # 0 = no ceiling
    # The most recent months count this many times an older one in every level /
    # feature / training weight (1 = no extra weight).
    weekly_recent_months: int = 6
    weekly_recent_weight: float = 2.0
    weekly_deviation_band: float = 0.5      # legacy symmetric band (unused by build)
    weekly_down_band: float = 0.30          # forecast held at >= 70% of the SKU level
    weekly_up_band: float = 2.00            # forecast allowed up to 300% of the SKU level
    # Never forecast zero for a SKU that has sold at least once in the history.
    weekly_min_units: int = 1
    # Borrow a little shape from same-category SKUs: each product's forecast/level
    # multiplier is shrunk this fraction toward its category's median multiplier
    # (0 = each SKU fully independent, 1 = every SKU uses the category multiplier).
    # ~0.2 lowers hold-out error a few points while keeping bias near zero.
    weekly_category_pool: float = 0.2
    # Pin the weekly model instead of auto-selecting it. "" = auto (bias-aware
    # pick); a model name forces that one for every product. "blend" (a weighted
    # mean of the calibrated ES-RNN-ratio and the XGBoost feature model) has the
    # lowest 1-week in-stock hold-out error while staying near-neutral bias,
    # which the safety uplift then lifts to the target small over-forecast.
    weekly_force_model: str = "blend"

    # Monthly branch "Item Statistics" exports for the demand forecast
    # (relative -> project root). Drop new <MONTH> <BRANCH> SALES.xlsx files here.
    sales_history_dir: str = "./data/sales_history"
    # Current stock-on-hand snapshot per branch, one file <BRANCHCODE>.xlsx
    inventory_dir: str = "./data/inventory"
    # Weekly branch sales exports for the weekly per-SKU forecast model
    # (one file per branch-week: "<BRANCH> DD-MM-YYYY to DD-MM-YYYY Sales.xlsx").
    weekly_sales_dir: str = "./data/weekly_sales"
    # Weekly stock-on-hand exports from Hansa, one file per branch-week (same
    # name convention). Used to spot stockout weeks: a week a SKU was out of
    # stock has its (suppressed) sales lifted to the SKU's in-stock level before
    # the models train, so the forecast reflects FULLY-STOCKED demand, not the
    # sales that a dry shelf allowed. Negative on-hand figures are read as 0.
    weekly_inventory_dir: str = "./data/weekly_inventory"
    # Lift stockout weeks to the in-stock level before training (fixes the
    # models' systematic under-prediction on products that keep running dry).
    weekly_unconstrain: bool = True
    weekly_family_borrow: bool = True
    weekly_family_borrow_share: float = 0.2     # share of the gap to sibling products closed (max)
    # Join the sales history of a re-coded product (old SKU stops as the new one
    # starts, identical name) into the live code before forecasting.
    weekly_merge_recoded: bool = True
    weekly_merge_scale: bool = True             # add the old history at the share the new code has taken over
    # Monthly-sourced data is scored on its last 3 months (hold-out + the
    # rolling-origin CV the offline trainer runs); weekly-sourced data keeps 1.
    weekly_holdout_periods: int = 3
    weekly_cv_origins: int = 3
    # When there is no inventory data for a (branch, SKU, week), still treat a
    # near-zero sales week as a stockout if the SKU normally sells most weeks and
    # that week is flanked by in-stock weeks.
    weekly_unconstrain_heuristic: bool = True
    # "monthly" (the real app's default): the weekly forecasting engine trains
    # and predicts from real monthly sales history (see monthly_sales.py) -
    # real weekly upload history is still too thin across most branches to
    # train on, so build() estimates a weekly rate from the monthly model's
    # prediction instead (divides by 4, same convention weekly_demand_estimate()
    # already used). "weekly" restores the original behaviour (WeeklySalesLine /
    # local weekly files) - the test suite pins this so its many synthetic-
    # weekly-data fixtures keep exercising the weekly-cadence code paths.
    weekly_data_source: str = "monthly"
    # The two torch models in the weekly comparison (each ~5-15 s to fit).
    # NeuralProphet uses a from-scratch implementation of its decomposition
    # (level + Fourier seasonality + AR-Net) because the PyPI package v0.9 does
    # not run against the installed torch/Lightning; set WMS_NEURALPROPHET_PKG=1
    # to try the real package once its deps are pinned to compatible versions.
    weekly_esrnn: bool = True
    weekly_neuralprophet: bool = True
    # ES-RNN "ratio" variant: predicts a multiplier in [ratio_lo, ratio_hi] on
    # a frequency-aware recent level (rather than an absolute value), trained to
    # sit just above actual so a branch is neither overstocked nor left short.
    weekly_esrnn_ratio: bool = True
    weekly_ratio_lo: float = 0.8
    weekly_ratio_hi: float = 1.5
    # Feature-based global models (see wms/analytics/weekly_ml.py):
    #   gbm  - XGBoost on ~24 engineered history/intermittency features, Tweedie
    #          objective. Pools information across every SKU; the strongest
    #          single model on the strict in-stock hold-out.
    #   lstm - a plain windowed LSTM sequence regressor (distinct from ES-RNN:
    #          no exponential-smoothing state).
    # Both train on the stockout-unconstrained matrix (fully-stocked demand).
    weekly_gbm: bool = True
    weekly_lgbm: bool = True
    weekly_lstm: bool = True
    # "Old Excel": the plain manual rule -> ceil(last-month total * 1.1) / 4 per
    # week, on RAW sales, no uplift/clamp. A hand-checkable baseline to pin (via
    # the Flow Analysis model picker) while there is not enough weekly history.
    weekly_old_excel: bool = True

    # Backorder processing: target days from OPEN to CLOSED
    backorder_sla_days: int = 14

    # Output directory for generated Excel / CSV / PNG (relative -> project root)
    output_dir: str = "./output"

    @property
    def resolved_database_url(self) -> str:
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            raw = url[len(prefix):]
            if raw and not raw.startswith("/") and ":" not in raw[:3]:
                return prefix + str((_ROOT / raw).resolve())
        return url

    @property
    def out(self) -> Path:
        p = Path(self.output_dir)
        if not p.is_absolute():
            p = _ROOT / p
        p = p.resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
