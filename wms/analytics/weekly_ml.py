"""Feature-based weekly forecasters: a gradient-boosted tree model (XGBoost) and
a windowed LSTM, both *global* (one model across every SKU series).

They are the "learn from engineered history features" counterpart to the ES-RNN
in ``weekly_neural``:

  gbm   - XGBoost regressor on ~24 features per (series, week): recent lags,
          rolling mean/std, intermittency stats (ADI, CV2, weeks-since-sale),
          the Croston rate and spike-damped level, a short trend, the panel
          week index, calendar month, and the branch / category ids. Tweedie
          objective (mass at zero + a non-negative right tail = intermittent
          demand). Direct 1-step, iterated for h>1.
  lstm  - a small multivariate LSTM over the last ``win`` weeks of
          [level-scaled units, had-sale flag] plus static [log1p(level),
          non-zero rate, branch, category], predicting next / level. Distinct
          from ES-RNN: no per-series exponential-smoothing state, a plain
          sequence regressor.

Each entry point is fitted on the training weeks only and returns an ``(S, h)``
non-negative array aligned to ``MAT``'s rows, or ``None`` if its library is
missing or the fit fails. Trained on whatever matrix ``weekly_forecast`` passes
(the stockout-unconstrained one), so they learn fully-stocked demand.
"""
from __future__ import annotations

import warnings

import numpy as np

_MINH = 4                       # need at least this many weeks of history for a row
_FEATS = [
    "lag1", "lag2", "lag3", "lag4", "lag5", "lag6",
    "roll4_mean", "roll4_max", "roll8_mean", "roll8_std",
    "mean_all", "std_all", "max_all",
    "nz_rate", "wk_since_sale", "adi", "cv2",
    "croston", "damped", "trend",
    "t_idx", "month", "branch_id", "cat_id",
]


def _helpers():
    """Small stats reused from weekly_forecast (imported lazily to avoid a
    circular import at module load)."""
    from wms.analytics.weekly_forecast import _damped_mean, f_croston
    return _damped_mean, f_croston


def _row(hist, t_idx, month, branch_id, cat_id, _damped, _croston):
    """One feature vector from a 1-D history array (units up to, not including,
    the week being predicted)."""
    h = np.asarray(hist, float)
    n = h.shape[0]

    def lag(k):
        return float(h[-k]) if n >= k else 0.0

    last4 = h[-4:] if n else np.zeros(1)
    last8 = h[-8:] if n else np.zeros(1)
    prev4 = h[-8:-4] if n >= 8 else np.zeros(1)
    nz = h[h > 0]
    if nz.size:
        last_sale = int(np.nonzero(h > 0)[0][-1])
        wk_since = float(n - 1 - last_sale)
        adi = n / nz.size
        cv2 = float(nz.var() / (nz.mean() ** 2 + 1e-9))
    else:
        wk_since, adi, cv2 = float(n), float(n), 0.0
    return [
        lag(1), lag(2), lag(3), lag(4), lag(5), lag(6),
        float(last4.mean()), float(last4.max() if last4.size else 0.0),
        float(last8.mean()), float(last8.std()),
        float(h.mean()) if n else 0.0, float(h.std()) if n else 0.0,
        float(h.max()) if n else 0.0,
        float((h > 0).mean()) if n else 0.0, wk_since, float(adi), cv2,
        float(_croston(h, 1)[0]) if n else 0.0,
        float(_damped(h)) if n else 0.0,
        float(last4.mean() - prev4.mean()),
        float(t_idx), float(month), float(branch_id), float(cat_id),
    ]


def _ids(keys, cats):
    """(branch_id array, category_id array) as small ints, plus the count of
    each, from the panel keys and a per-series category-label array."""
    keys = keys if keys is not None else [("_", str(i)) for i in range(len(cats))]
    cats = (np.asarray(cats) if cats is not None
            else np.zeros(len(keys), dtype=object))
    b_map, c_map = {}, {}
    b_of = np.array([b_map.setdefault(k[0], len(b_map)) for k in keys])
    c_of = np.array([c_map.setdefault(cats[i], len(c_map)) for i in range(len(keys))])
    return b_of, c_of, max(1, len(b_map)), max(1, len(c_map))


def _months(weeks, W):
    import pandas as pd
    out = []
    for t in range(W):
        try:
            out.append(pd.Timestamp(weeks[t]).month)
        except Exception:                                    # noqa: BLE001
            out.append(0)
    return out


def _design(MAT, weeks, tr_end, keys, cats):
    """Shared design matrix for the tree models: every (series, week) row in
    weeks [_MINH, tr_end) with its feature vector and target, plus the id/month
    lookups needed to roll features forward at predict time."""
    _damped, _croston = _helpers()
    S, W = MAT.shape
    b_of, c_of, _nb, _nc = _ids(keys, cats if cats is not None
                                else np.zeros(S, dtype=object))
    months = _months(weeks, W)
    X, y = [], []
    for i in range(S):
        hi = MAT[i]
        for t in range(_MINH, tr_end):
            X.append(_row(hi[:t], t, months[t] if t < W else 0,
                          b_of[i], c_of[i], _damped, _croston))
            y.append(hi[t])
    return (np.asarray(X, np.float32), np.asarray(y, np.float32),
            b_of, c_of, months, _damped, _croston)


def _iterate(booster, predict, MAT, tr_end, h, b_of, c_of, months,
             _damped, _croston):
    """Roll a fitted 1-step booster forward h weeks, feeding predictions back."""
    S, W = MAT.shape
    hist = [MAT[i, :tr_end].tolist() for i in range(S)]
    out = np.zeros((S, h), np.float32)
    for step in range(h):
        t = tr_end + step
        mo = months[t] if t < W else (months[-1] if months else 0)
        feats = np.asarray(
            [_row(hist[i], t, mo, b_of[i], c_of[i], _damped, _croston)
             for i in range(S)], np.float32)
        pred = np.clip(predict(booster, feats), 0, None)
        out[:, step] = pred
        for i in range(S):
            hist[i].append(float(pred[i]))
    return out


def gbm_forecast(MAT, weeks, tr_end, h, *, keys=None, cats=None,
                 n_estimators=450, max_depth=5, lr=0.05, seed=0,
                 tweedie_power=1.3, model_out=None, model_in=None):
    """XGBoost global 1-step regressor, iterated for h>1. Native
    ``xgboost.train`` / ``Booster`` API (no scikit-learn dependency)."""
    try:
        import xgboost as xgb
    except Exception:                                        # noqa: BLE001
        return None
    try:
        MAT = np.clip(np.asarray(MAT, float), 0, None)
        S, W = MAT.shape
        if S == 0 or tr_end < _MINH + 1:
            return None
        X, y, b_of, c_of, months, _damped, _croston = _design(
            MAT, weeks, tr_end, keys, cats)

        booster = None
        if model_in is not None:
            try:
                booster = xgb.Booster()
                booster.load_model(model_in)
            except Exception:                                # noqa: BLE001
                booster = None
        if booster is None:
            if len(X) < 50:
                return None
            params = {
                "objective": "reg:tweedie", "tweedie_variance_power": tweedie_power,
                "eta": lr, "max_depth": max_depth, "subsample": 0.85,
                "colsample_bytree": 0.85, "min_child_weight": 5,
                "lambda": 1.0, "seed": seed, "nthread": 0,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = xgb.train(params, xgb.DMatrix(
                    X, label=y, feature_names=_FEATS), num_boost_round=n_estimators)
            if model_out is not None:
                try:
                    booster.save_model(model_out)
                except Exception:                            # noqa: BLE001
                    pass

        def _pred(b, feats):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return b.predict(xgb.DMatrix(feats, feature_names=_FEATS))

        return _iterate(booster, _pred, MAT, tr_end, h, b_of, c_of, months,
                        _damped, _croston)
    except Exception as e:                                   # noqa: BLE001
        warnings.warn(f"gbm_forecast failed: {e}")
        return None


def lgbm_forecast(MAT, weeks, tr_end, h, *, keys=None, cats=None,
                  n_estimators=500, num_leaves=31, lr=0.04, seed=0,
                  tweedie_power=1.3, model_out=None, model_in=None):
    """LightGBM global 1-step regressor, iterated for h>1. Same features as
    ``gbm`` but a leaf-wise tree grower and slightly deeper ensemble - usually
    decorrelated enough from XGBoost to help a blend. Native ``lgb.train`` API."""
    try:
        import lightgbm as lgb
    except Exception:                                        # noqa: BLE001
        return None
    try:
        MAT = np.clip(np.asarray(MAT, float), 0, None)
        S, W = MAT.shape
        if S == 0 or tr_end < _MINH + 1:
            return None
        X, y, b_of, c_of, months, _damped, _croston = _design(
            MAT, weeks, tr_end, keys, cats)

        booster = None
        if model_in is not None:
            try:
                booster = lgb.Booster(model_file=model_in)
            except Exception:                                # noqa: BLE001
                booster = None
        if booster is None:
            if len(X) < 50:
                return None
            params = {
                "objective": "tweedie", "tweedie_variance_power": tweedie_power,
                "learning_rate": lr, "num_leaves": num_leaves,
                "min_child_samples": 20, "subsample": 0.85, "subsample_freq": 1,
                "colsample_bytree": 0.85, "reg_lambda": 1.0, "seed": seed,
                "num_threads": 0, "verbosity": -1,
            }
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                booster = lgb.train(params, lgb.Dataset(
                    X, label=y, feature_name=list(_FEATS)),
                    num_boost_round=n_estimators)
            if model_out is not None:
                try:
                    booster.save_model(model_out)
                except Exception:                            # noqa: BLE001
                    pass

        return _iterate(booster, lambda b, f: b.predict(f), MAT, tr_end, h,
                        b_of, c_of, months, _damped, _croston)
    except Exception as e:                                   # noqa: BLE001
        warnings.warn(f"lgbm_forecast failed: {e}")
        return None


def lstm_forecast(MAT, weeks, tr_end, h, *, keys=None, cats=None, win=8,
                  hidden=24, epochs=90, lr=6e-3, seed=0, under_w=1.4,
                  lo=0.35, hi=3.0):
    """Small global windowed LSTM: last ``win`` weeks of [units/level, had-sale]
    + static [log1p(level), nz_rate, branch, category] -> a multiplier on the
    level, bounded to ``[lo, hi]`` (lo > 0 so a live SKU is never forecast to
    zero) and trained to lean above actual (``under_w``)."""
    try:
        import torch
        import torch.nn as nn
    except Exception:                                        # noqa: BLE001
        return None
    try:
        _damped, _croston = _helpers()
        torch.manual_seed(seed)
        np.random.seed(seed)
        MAT = np.clip(np.asarray(MAT, np.float32), 0, None)
        S, W = MAT.shape
        if S == 0 or tr_end < win + 2:
            return None
        b_of, c_of, nb, nc = _ids(keys, cats if cats is not None
                                  else np.zeros(S, dtype=object))
        lvl = np.array([max(1.0, _damped(MAT[i, :tr_end])) for i in range(S)],
                       np.float32)
        nz_rate = (MAT[:, :tr_end] > 0).mean(1).astype(np.float32)

        Xs, Bs, Cs, St, Y = [], [], [], [], []
        for i in range(S):
            li = lvl[i]
            for t in range(win, tr_end):
                seg = MAT[i, t - win:t]
                Xs.append(np.stack([seg / li, (seg > 0).astype(np.float32)], 1))
                Bs.append(b_of[i])
                Cs.append(c_of[i])
                St.append([np.log1p(li), nz_rate[i]])
                Y.append(MAT[i, t] / li)
        if len(Xs) < 50:
            return None
        Xs = torch.tensor(np.asarray(Xs, np.float32))
        Bs = torch.tensor(np.asarray(Bs, np.int64))
        Cs = torch.tensor(np.asarray(Cs, np.int64))
        St = torch.tensor(np.asarray(St, np.float32))
        Y = torch.tensor(np.asarray(Y, np.float32)).clamp(lo, hi)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(2, hidden, batch_first=True)
                self.be = nn.Embedding(nb, 4)
                self.ce = nn.Embedding(nc, 4)
                self.head = nn.Sequential(
                    nn.Linear(hidden + 4 + 4 + 2, 32), nn.ReLU(),
                    nn.Linear(32, 1))

            def forward(self, x, b, c, s):
                o, _ = self.rnn(x)
                z = torch.cat([o[:, -1], self.be(b), self.ce(c), s], 1)
                return self.head(z).squeeze(1)

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        n = Xs.shape[0]
        uw = torch.tensor(under_w)
        one = torch.tensor(1.0)
        for _ep in range(epochs):
            perm = torch.randperm(n)
            for j in range(0, n, 4096):
                b = perm[j:j + 4096]
                opt.zero_grad()
                p = net(Xs[b], Bs[b], Cs[b], St[b]).clamp(min=0.0)
                err = p - Y[b]
                w = torch.where(err < 0, uw, one)
                (w * err.abs()).mean().backward()
                opt.step()

        net.eval()
        hist = MAT[:, :tr_end].astype(np.float32).copy()
        out = np.zeros((S, h), np.float32)
        b_t = torch.tensor(b_of)
        c_t = torch.tensor(c_of)
        with torch.no_grad():
            for step in range(h):
                segs = np.stack([
                    np.stack([hist[i, -win:] / lvl[i],
                              (hist[i, -win:] > 0).astype(np.float32)], 1)
                    for i in range(S)], 0)
                x = torch.tensor(segs.astype(np.float32))
                st = torch.tensor(np.stack([np.log1p(lvl), nz_rate], 1)
                                  .astype(np.float32))
                r = net(x, b_t, c_t, st).clamp(lo, hi).numpy()
                pred = np.clip(r * lvl, 0, None)
                out[:, step] = pred
                hist = np.concatenate([hist, pred[:, None]], 1)
        return out
    except Exception as e:                                   # noqa: BLE001
        warnings.warn(f"lstm_forecast failed: {e}")
        return None
