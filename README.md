# Mineazy WMS — Backorder & Branch-Demand system

A browser web app for staff, a FastAPI JSON API, and an interactive console — all
over one shared service/analytics layer.

Focus: the **backorder processing flow** (a 9-stage procurement workflow), deep
**fulfilment & flow analytics**, **branch sales analysis** and demand
**forecasting** to anticipate branch orders, pandas / Matplotlib / Seaborn
visualisation, and **Excel + CSV export**.

> There is **no stock ledger, multi-stage stock tracking, adjustments, cycle
> counting, ASN, or low-stock alerts**. The Inventory page is a per-branch
> stock-on-hand snapshot (upload a spreadsheet, replaces the branch's balance)
> and a read-only summary/search view — not a movement or transaction log.
> Delivery notes are kept as the requested-vs-sent source document only.

| Layer | Tech |
|---|---|
| Web app (primary UI) | FastAPI + Jinja2 server-rendered pages, **login + roles**, no build step |
| JSON API | FastAPI (Swagger at `/api/docs`) |
| Console | `questionary` + `rich` (`python -m wms.console`) |
| ORM | SQLAlchemy 2.x |
| Database | **MySQL** (`pymysql`); SQLite fallback for zero-setup |
| Data / analysis | pandas, numpy · Matplotlib + Seaborn charts |
| Export | openpyxl / XlsxWriter (Excel), stdlib `csv` |
| Auth | session cookie + bcrypt; roles: admin · controller · clerk · branch · analyst |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt      # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux

python -m wms.scripts.seed        # demo data incl. Stock-Movement doc 26503244
python -m uvicorn wms.api.main:app --reload --port 8000
```

Then open **http://127.0.0.1:8000** and sign in. Demo logins (password `wms1234`):

| user | role | can |
|---|---|---|
| `admin` | System Administrator | everything |
| `controller` | Procurement Controller | manage back orders (advance stages / cancel) |
| `clerk` | Branch / Order Clerk | enter back orders and delivery notes |
| `branch` | Branch User | view only |
| `analyst` | Reporting Analyst | view + exports |

```bash
python -m wms.console            # interactive console (alternative to the web UI)
python -m wms.console --demo     # non-interactive read-only tour
python -m pytest                 # 27 tests
```

### Importing a real catalogue

```bash
python -m wms.scripts.import_stock_list "STOCK LIST.xlsx"
python -m wms.scripts.import_stock_list "STOCK LIST.xlsx" --fresh   # wipe demo data first
```

Columns: `Item No`, `Name`, `Group` (→ category, expanded to a readable label).
Products are upserted by `Item No`.

Swagger for the JSON API is at **/api/docs**.
MySQL: set `DATABASE_URL=mysql+pymysql://…` in `.env` (see `.env.example`), or `docker compose up`.
Generated workbooks / CSVs / PNGs are written to `./output/`.

---

## Backorders — the processing flow

A **back order** is a header (`BackOrder`) + item lines (`BackOrderItem`) that moves
through a procurement workflow; every transition writes a `BackOrderEvent` (the raw
material for cycle-time analysis). Stages:

```
SUBMITTED → WAREHOUSE_REVIEW → PROCUREMENT_NEEDED → REQUISITION → PO_ISSUED
          → GOODS_RECEIVED → READY_TO_ALLOCATE → DISPATCHED → CLOSED     (+ CANCELLED)
Warehouse Review may skip straight to Ready to Allocate (stock found).
```

Item quantities are recorded stage-by-stage: `qty_ordered → approved → on_po →
received → allocated → dispatched → fulfilled`. Workflow-only: transitions record
quantities + timestamps.

**Two entry modules:**
- **Manual** — `create_back_order(branch, items=[{sku, qty}], priority, notes)`
  (`services/backorder_entry.py`).
- **From a delivery note** — entering a DN records the `sent` qty as branch sales
  and, for every line where `requested > sent`, auto-creates one back order at
  `SUBMITTED` (`backorder_qty = requested − sent`, blank sent = 0).

The stage machine lives in `services/backorder_stages.py` — `advance(bo_no,
to_stage, …)` guards transitions and requires the stage's payload (PO number,
requisition number, per-item quantity).

**Analysis module** (`analytics/backorder_flow.py`) — overall **and** by branch:
- **Fulfilment:** fill rate (qty & line), open/closed/cancelled, mean fulfilment %,
  outstanding value, mean/median lead time, overdue count, on-time-close rate.
- **Flow:** stage funnel (where back orders sit now), **days in each stage**,
  **bottleneck stage**, aging (0-7 / 8-14 / 15-30 / 30+), aging × stage heatmap.
- **By branch:** count, fill %, mean lead time, overdue, outstanding value, top items.
- **Backorders vs sales by branch:** demand met % = sales / (sales + backordered).
- **Trend:** weekly raised / closed / outstanding + fill-rate.

**Export:** `backorder_flow_workbook()` — 12 sheets (Summary, Active Back Orders,
Items, Stage Funnel, Cycle Times, Aging, Aging by Stage, By Branch,
Top Items Outstanding, Branch BO vs Sales, Trend, Events).
`backorder_flow_pack()` renders 7 Matplotlib/Seaborn charts.

The delivery-note fill workbook (`backorder_workbook()`, requested-vs-sent view) is
still available as the upstream signal.

## Branch sales analysis, forecasting & allocation

- **Overall / per-branch KPIs**, branch comparison, **branch sales summary**
  (qty, value, distinct SKUs, weekly average, trend %), ABC by sales value,
  weekly sales trend.
- **Demand forecast** — recency-weighted average daily demand, `demand_std`,
  safety stock, reorder point, target level.
- **Suggested branch orders** — order-up-to-target: `suggested = ceil(target_level)`
  where `target_level = avg·(lead_time + review_period) + z·std·√lead_time`.
  (No stock on-hand is tracked, so there is no `− on_hand` term.)
- **Allocation** — you supply an available quantity; `allocate_product()` splits it
  across branches: up to each reorder point in demand order, then the remainder
  proportional to demand share.

## Anticipating branch orders — the demand model (built separately)

`python -m wms.scripts.export_training_data` writes a tidy `(branch, product, ISO-week)`
table (target = next week's demand) with leak-free features: demand lags 1–8,
rolling mean/std (4/8/13w), trend, calendar (week/month, sin/cos of week-of-year,
month-end), weeks-since-last-sale, zero-share (intermittency), unit price & band,
category, and recent-backorder signals (`bo_qty`, `bo_count`, `bo_qty_roll_4`).

**Steps to train it:**
1. **Target & grain:** `(branch, product, week)` → next-week demand (regression);
   optionally a second binary target "backordered within N days".
2. **Assemble data:** run `export_training_data` (uses sales history + back-order signals).
3. **Features:** already produced by the script — extend with supplier lead time,
   promotions/price changes, known future demand.
4. **Split by time:** earliest ~70 % train, next ~15 % validate, last ~15 % test;
   rolling-origin backtest. Never shuffle.
5. **Baselines:** naive, seasonal-naive, 4-week MA, Croston/SBA for intermittent SKUs.
6. **Model:** LightGBM/XGBoost with Poisson/Tweedie (or quantile) loss; two-stage
   (P(any demand) × size) for very intermittent items.
7. **Tune** with time-series CV; log runs.
8. **Evaluate** with MAE / WAPE / MASE / bias per branch & category, then **simulate**
   feeding forecasts into the reorder-point logic (fill rate, backorders raised)
   vs the current heuristic.
9. **Order qty:** `target_level = forecast·(LT + review) + z·forecast_std`;
   `suggested = ceil(target_level)`.
10. **Wire in:** save with `joblib`; expose `predict(branch_id, product_id, as_of)
    → (mean, std)`; call it behind a flag in `wms/analytics/forecast.py`, heuristic
    as fallback.
11. **Monitor & retrain:** track weekly error (drift) + realised fill rate; retrain
    monthly or on drift; champion/challenger.

## Reporting & exports

- **Audit trail** — every mutating action with `user_id` (`GET /api/reports/audit`).
- **Excel** — backorder-flow workbook (12 sheets), delivery-note fill, branch sales,
  suggested orders, allocation plan, branch statistics.
- **CSV** — back orders, sales history (for the model).
- **Charts (PNG)** — dashboard pack (sales by branch, ABC Pareto, DN fill-rate,
  top short items, DN trend, ageing, branch×item heatmap) and the
  backorder-flow pack (stage funnel, days-in-stage, fulfilment by branch,
  aging heatmap, backorders vs sales, lead-time histogram, weekly flow).

---

## Web app

ERP-style shell: white sidebar with an indigo active pill, light-grey canvas, white
cards, and the signed-in **account + role shown top-right**. Four nav items:

| Page | What it does |
|---|---|
| **New Back Order** | **Upload a CSV/Excel** of the stock-movement document → the header (Stock Movement ID, Branch, Date) and line items are parsed and the form is pre-filled; or key it in. A blank *dispatched* cell = nothing sent, so the whole line is backordered. Saving records the dispatched qty as branch sales and raises one back order for the shortfall. |
| **Active Back Orders** | the grid — `Order # · Stock Movement · Branch · Stage · Items · Fulfillment · Status · Actions`, with an **All Stages** filter, branch filter and search. A link opens **Flow analysis** (fulfilment, funnel, cycle times, aging, by branch, backorders-vs-sales, trend). |
| **Sales & Forecasting** | statistics · weekly sales trend · ABC by sales value · forecast + suggested branch orders · allocation plan. |
| **Reports & Exports** | Excel workbooks, CSV extracts for the demand model, chart packs. |

The **New Back Order** upload parser (`services/doc_import.py`) recognises a header
row with `Item No` + a `Req. Qty` column, maps `Sent Qty` / `Dispatched` (blank →
0), and best-effort reads the Stock Movement number, branch and date from the
document header.

Every action is written to the audit trail with the logged-in user. Buttons a
role can't use are hidden, and the route is enforced server-side too.

## Console menu

```
Main Menu
 ├─ Dashboard & KPIs             overall snapshot · backorder-flow snapshot ·
 │                               per-branch · branch sales summary
 ├─ Backorders                   active back orders (stage filter) · enter (manual) ·
 │                               advance stage · enter delivery note · import CSV ·
 │                               flow analysis OVERALL / BY BRANCH · backorders vs sales ·
 │                               delivery-note fill analysis · Excel exports · chart pack
 ├─ Sales, Forecasting …         overall / per-branch stats · comparison · branch sales ·
 │                               ABC · weekly sales trend · forecast + suggested orders ·
 │                               allocate one product · suggested orders per product×branch
 └─ Reports & Exports            audit trail · Excel workbooks · CSV · chart packs
```

## Layout

```
wms/
├── wms/
│   ├── config.py db.py enums.py models.py audit.py errors.py security.py
│   ├── services/        catalog · backorders (delivery notes) · backorder_entry ·
│   │                    backorder_stages · sales
│   ├── analytics/       loaders · backorders (DN fill) · backorder_flow ·
│   │                    statistics · forecast · allocation
│   ├── viz/             theme + Matplotlib/Seaborn charts + backorder_flow_pack
│   ├── exports/         excel.py · csv_export.py
│   ├── api/             FastAPI app + JSON routes (outbound/back-orders, analytics, reports)
│   ├── web/             browser app: deps.py (auth/RBAC) · routes.py · templates/
│   ├── console/         interactive menu (app.py, ui.py, menus/)
│   └── scripts/         seed.py · sample_data.py · import_stock_list.py · export_training_data.py
├── tests/               27 tests (entry · flow state machine · analytics · API · web · RBAC)
├── alembic/             migration scaffold (URL + metadata wired)
├── output/              generated .xlsx / .csv / .png
├── requirements.txt  docker-compose.yml  Dockerfile  run.ps1  run.sh
```

## API surface (selected)

| Method & path | Purpose |
|---|---|
| `POST /api/outbound/delivery-notes` · `GET .../{no}` | enter a delivery note (→ sales + back order) |
| `POST /api/outbound/back-orders` · `GET /api/outbound/back-orders?stage=&q=` | create / list back orders |
| `GET  /api/outbound/back-orders/{no}` · `/next-stages` · `POST .../advance` · `/cancel` | back-order flow |
| `GET  /api/outbound/back-orders-analysis` | fulfilment · funnel · cycle times · aging · by branch · vs sales · trend |
| `GET  /api/analytics/overall` · `/branch/{id}` · `/branch-sales` · `/abc` · `/sales-trend` | statistics + sales analysis |
| `GET  /api/analytics/backorders` | delivery-note fill analysis (requested vs sent) |
| `GET  /api/analytics/forecast` · `/suggested-orders` · `/allocation-plan` · `POST /allocate` | forecasting + allocation |
| `GET  /api/reports/audit` | audit trail (entity / user / action filters) |
| `GET  /api/reports/{backorder-flow,backorders,branch-sales,suggested-orders,allocation-plan,branch-stats}.xlsx` | Excel |
| `GET  /api/reports/{back-orders,sales}.csv` | CSV for the demand model |

The acting user for audit tracking is the `X-Actor` request header (default `admin`).

## Deploying (Render + Aiven MySQL, no credit card)

Two free services, no payment method required on either:

1. **Database - Aiven, free MySQL plan** ([aiven.io/free-mysql-database](https://aiven.io/free-mysql-database)).
   Create a service, then from its Overview page grab the **Host, Port, User,
   Password, Database name**, and download the **CA Certificate** (Aiven's
   free MySQL requires TLS). The service auto-powers-off after a long stretch
   of inactivity - Aiven emails a warning first, and a manual restart from
   their dashboard brings it straight back with the data intact.

2. **App - Render, free web service** ([render.com](https://render.com)).
   Push this repo to GitHub, then in Render: **New → Blueprint**, point it at
   the repo - it reads [`render.yaml`](render.yaml) and creates the service.
   Before (or right after) the first deploy:
   - **Environment → Secret Files → Add Secret File**: name it `ca.pem`,
     paste in the CA certificate content from step 1 (Render serves it to the
     container at `/etc/secrets/ca.pem`, which is what `DB_SSL_CA` in
     `render.yaml` already points at).
   - **Environment → DATABASE_URL**: set to
     `mysql+pymysql://<user>:<password>@<host>:<port>/<database>` using the
     Aiven values from step 1.
   - `SECRET_KEY` is filled in for you (random, generated by Render).

   Render builds the [`Dockerfile`](Dockerfile) (installs the Tesseract OCR
   engine + Python deps) and starts the app; `/api/health` is the health
   check.

**First login - your call:** a brand-new Aiven database has no tables yet;
the app creates them automatically on first boot (`init_db()`), but there are
no user accounts until something seeds one. `python -m wms.scripts.seed`
creates the demo users/catalogue/branches (`admin` / `wms1234`, etc.) - **but
it starts with `Base.metadata.drop_all()`, wiping the target database first**,
so only run it against a database you're fine starting empty on. If you'd
rather carry over your existing local data, migrate/export it into the Aiven
database directly instead of seeding.

**Known free-tier limits:** Render's free-tier disk is ephemeral (wiped on
every restart/redeploy) - fine for `OUTPUT_DIR` (regenerated exports) but it
means anything dropped into `data/sales_history`, `data/inventory`, etc.
needs re-uploading after a restart, since that isn't backed by the database.
The free web service also spins down after 15 minutes idle, so the first
request after a quiet spell is slow to wake it up. The weekly demand model's
neural candidates (torch-based) are disabled by default in `render.yaml` to
fit the free plan's 512MB RAM - see the comments in that file.
