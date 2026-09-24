"""Neural weekly forecasters — ES-RNN and NeuralProphet.

Both are *global* models (one model across every SKU series), fitted on the
training weeks only and asked for an ``h``-step-ahead forecast made once (the
held-out weeks are never shown to them).

Each entry point returns an ``(S, h)`` numpy array aligned to ``MAT``'s rows, or
``None`` if torch is unavailable or training fails.
"""
from __future__ import annotations

import warnings

import numpy as np


# ----------------------------------------------------------------- ES-RNN
def esrnn_forecast(MAT: np.ndarray, tr_end: int, h: int, *,
                   win: int = 4, hidden: int = 16, epochs: int = 40,
                   lr: float = 7e-3, seed: int = 0):
    """Smyl-style ES-RNN (non-seasonal): a per-series exponential-smoothing level
    with a learnable smoothing constant, feeding a shared LSTM that predicts the
    level-normalised path h steps ahead. Trained by SGD over all series."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = np.clip(MAT[:, :tr_end].astype(np.float32), 0, None)
        S, T = train.shape
        if T < win + h + 1 or S == 0:
            return None
        Y = torch.tensor(train)

        a = torch.zeros(S, requires_grad=True)            # alpha = sigmoid(a)
        seed_lvl = Y[:, :max(1, win)].mean(1).clamp(min=1e-3)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(win, hidden, batch_first=True)
                self.head = nn.Linear(hidden, h)

            def forward(self, x):
                o, _ = self.rnn(x)
                return self.head(o[:, -1])

        net = Net()
        opt = torch.optim.Adam(list(net.parameters()) + [a], lr=lr, weight_decay=1e-4)

        def levels(alpha):
            lv = seed_lvl.clone()
            cols = []
            for t in range(T):
                lv = alpha * Y[:, t] + (1 - alpha) * lv
                cols.append(lv)
            return torch.stack(cols, 1).clamp(min=1e-3)

        starts = list(range(win - 1, T - h))
        from wms.analytics.weekly_forecast import recency_weights
        _rw = recency_weights(T)              # recent months count more
        for _ in range(epochs):
            opt.zero_grad()
            alpha = torch.sigmoid(a)
            lv = levels(alpha)
            loss = 0.0
            wsum = 0.0
            for t in starts:
                base = lv[:, t].unsqueeze(1)
                xin = (Y[:, t - win + 1: t + 1] / base).log1p().unsqueeze(1)
                tgt = Y[:, t + 1: t + 1 + h]
                pred = net(xin).expm1() * base
                tw = float(_rw[min(t + h, T - 1)])
                loss = loss + tw * (pred - tgt).abs().mean()
                wsum += tw
            loss = loss / max(wsum, 1e-9)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(net.parameters()) + [a], 5.0)
            opt.step()

        with torch.no_grad():
            alpha = torch.sigmoid(a)
            lv = levels(alpha)
            base = lv[:, -1].unsqueeze(1)
            xin = (Y[:, T - win: T] / base).log1p().unsqueeze(1)
            out = (net(xin).expm1() * base).clamp(min=0).cpu().numpy()
        return out.astype(float)
    except Exception as e:                                # noqa: BLE001
        warnings.warn(f"ES-RNN failed: {e}")
        return None


# ------------------------------------------------------- ES-RNN (multiplier)
def _croston_rate(y: np.ndarray, a: float = 0.1) -> float:
    """SBA (bias-corrected Croston) mean weekly demand for one series."""
    nz = np.nonzero(y)[0]
    if len(nz) == 0:
        return 0.0
    z, p, q = float(y[nz[0]]), float(nz[0] + 1), 1.0
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = a * y[t] + (1 - a) * z
            p = a * q + (1 - a) * p
            q = 1.0
        else:
            q += 1.0
    return max(0.0, (1 - a / 2) * z / p)


def esrnn_ratio_forecast(MAT: np.ndarray, tr_end: int, h: int, *, base=None,
                         win: int = 4, hidden: int = 16, epochs: int = 70,
                         lr: float = 8e-3, seed: int = 0,
                         lo: float = 0.8, hi: float = 1.5,
                         under_w: float = 1.6, zero_w: float = 0.4,
                         checkpoint=None, return_state: bool = False,
                         val_weeks: int = 0, patience: int = 40):
    """ES-RNN variant that predicts a MULTIPLIER in ``[lo, hi]`` (default
    0.8-1.5) on the SKU's recent-demand level instead of an absolute value.

    ``checkpoint``  a dict ``{"net_state", "hp"}`` from a previous train. When
        given and its hyper-params match, the shared LSTM+head weights are loaded
        and FROZEN and only each series' exponential-smoothing ``alpha`` is
        re-fitted (fast: seconds, not a minute) — this is how a model trained
        thoroughly offline is reused for cheap inference and still adapts when a
        branch/SKU is added.
    ``return_state``  also return ``(forecast, state)`` so the caller can persist
        the trained network as a checkpoint.
    ``val_weeks``  hold this many of the most-recent training windows out as a
        validation set; keep the epoch with the best validation loss (early
        stopping with ``patience``). Used by the offline trainer for a proper fit.

        forecast = base * r
          base = the SKU's RECENT weekly level (mean of the last few weeks),
                 floored at its long-run Croston/SBA rate so a SKU that just
                 went quiet keeps a baseline. A product that has recently ramped
                 up is scaled from where it is now, not an all-time average.
          r    = lo + (hi - lo) * sigmoid(net(...))            (in [lo, hi])

    ``net`` is a shared LSTM over the last ``win`` level-normalised weeks plus a
    handful of whole-history summary stats (volatility, how often the SKU sells,
    recent vs long-run average, spike-proneness, last-week vs mean). It decides,
    per series, where in ``[lo, hi]`` to sit — near ``hi`` for a SKU that is
    accelerating, near ``lo`` for one that is fading or that just spiked and is
    likely to revert. Trained by SGD across EVERY series (both branches) on the
    training weeks only.

    The loss is level-normalised absolute error with an under-forecast weighted
    ``under_w`` x an over-forecast (and zero-demand weeks down-weighted to
    ``zero_w`` so the multiplier is learned from the weeks that carry signal),
    so the pick lands close to actual yet leans just enough above it that a
    branch is not left short — without piling on excess stock.
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train = np.clip(MAT[:, :tr_end].astype(np.float32), 0, None)
        S, T = train.shape
        if T < win + h + 1 or S == 0:
            return None
        Y = torch.tensor(train)

        if base is None:
            # RECENT level first: the mean of the last few weeks, so a SKU that
            # has just ramped up (0 -> 200 -> 400) is scaled from where it is
            # now. Floor it at the long-run Croston rate so a SKU that recently
            # went quiet still carries its intermittent-demand baseline — unless
            # it has gone 6+ STRAIGHT weeks with nothing, in which case Croston's
            # rate can still be inflated by one huge one-off order long ago, and
            # resurrecting a now-dead SKU from that is wrong; use the recent
            # level alone (the caller's never-zero floor still applies on top).
            recent = train[:, -min(4, T):].mean(1)
            rate = np.array([_croston_rate(train[i]) for i in range(S)], np.float32)
            dw = min(6, T)
            dead = ~train[:, -dw:].any(axis=1) if dw else np.zeros(S, bool)
            base = np.where(dead, recent, np.maximum(recent, rate))
        B = torch.tensor(np.clip(np.asarray(base, np.float32), 1e-3, None))  # (S,)

        mean = Y.mean(1).clamp(min=1e-3)
        feats = torch.stack([
            (Y.std(1) / mean).clamp(0, 5),
            (Y > 0).float().mean(1),
            (Y[:, -4:].mean(1) / mean).clamp(0, 5),
            (Y.max(1).values / mean).clamp(0, 20) / 20.0,
            (Y[:, -1] / mean).clamp(0, 8) / 8.0,
        ], 1)                                                          # (S, 5)

        a = torch.zeros(S, requires_grad=True)             # alpha = sigmoid(a)
        seed_lvl = Y[:, :max(1, win)].mean(1).clamp(min=1e-3)

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(1, hidden, batch_first=True)
                self.head = nn.Linear(hidden + feats.shape[1], h)

            def forward(self, seq, extra):
                o, _ = self.rnn(seq)
                return self.head(torch.cat([o[:, -1], extra], 1))

        net = Net()
        n_feats = feats.shape[1]
        _hp = {"hidden": hidden, "win": win, "h": h, "n_feats": n_feats}
        warm = bool(checkpoint) and checkpoint.get("hp") == _hp
        if warm:
            net.load_state_dict({k: torch.as_tensor(v)
                                 for k, v in checkpoint["net_state"].items()})

        if warm:                                    # reuse net, only fit alpha
            for p in net.parameters():
                p.requires_grad_(False)
            opt = torch.optim.Adam([a], lr=lr * 2)
            fit_epochs = min(epochs, 15)
        else:
            opt = torch.optim.Adam(list(net.parameters()) + [a], lr=lr,
                                   weight_decay=1e-4)
            fit_epochs = epochs

        def levels(alpha):
            lv = seed_lvl.clone()
            cols = []
            for t in range(T):
                lv = alpha * Y[:, t] + (1 - alpha) * lv
                cols.append(lv)
            return torch.stack(cols, 1).clamp(min=1e-3)

        def ratio(z):
            return lo + (hi - lo) * torch.sigmoid(z)

        w_over, w_under = torch.as_tensor(1.0), torch.as_tensor(under_w)
        w_zero, w_pos = torch.as_tensor(zero_w), torch.as_tensor(1.0)
        all_starts = list(range(win - 1, T - h))
        vw = max(0, min(val_weeks, len(all_starts) - 1)) if not warm else 0
        tr_starts = all_starts[:len(all_starts) - vw] if vw else all_starts
        va_starts = all_starts[len(all_starts) - vw:] if vw else []

        # windows whose targets fall in the most recent months count more
        from wms.analytics.weekly_forecast import recency_weights
        _rw = recency_weights(T)

        def _loss(lv, starts):
            tot = 0.0
            wsum = 0.0
            for t in starts:
                base_lv = lv[:, t].unsqueeze(1)
                seq = (Y[:, t - win + 1: t + 1] / base_lv).log1p().unsqueeze(-1)
                adj = (lv[:, t] / lv[:, -1]).clamp(0.3, 3.0)
                base_t = (B * adj).clamp(min=1e-3).unsqueeze(1)
                tgt = Y[:, t + 1: t + 1 + h]
                err = base_t * ratio(net(seq, feats)) - tgt
                w = torch.where(err < 0, w_under, w_over) * \
                    torch.where(tgt > 0, w_pos, w_zero)
                tw = float(_rw[min(t + h, T - 1)])
                tot = tot + tw * (w * err.abs() / mean.unsqueeze(1)).mean()
                wsum += tw
            return tot / max(wsum, 1e-9)

        best_val, best_state, bad = float("inf"), None, 0
        params = ([a] if warm else list(net.parameters()) + [a])
        for ep in range(fit_epochs):
            opt.zero_grad()
            lv = levels(torch.sigmoid(a))
            _loss(lv, tr_starts).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            if vw:
                with torch.no_grad():
                    v = float(_loss(levels(torch.sigmoid(a)), va_starts))
                if v < best_val - 1e-4:
                    best_val, bad = v, 0
                    best_state = ({k: t.clone() for k, t in net.state_dict().items()},
                                  a.detach().clone())
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if vw and best_state is not None:
            net.load_state_dict(best_state[0])
            with torch.no_grad():
                a.copy_(best_state[1])

        with torch.no_grad():
            lv = levels(torch.sigmoid(a))
            base_lv = lv[:, -1].unsqueeze(1)
            seq = (Y[:, T - win: T] / base_lv).log1p().unsqueeze(-1)
            out = (B.unsqueeze(1) * ratio(net(seq, feats))).clamp(min=0).cpu().numpy()
        out = out.astype(float)
        if return_state:
            state = {"hp": _hp, "net_state":
                     {k: v.cpu().numpy() for k, v in net.state_dict().items()}}
            return out, state
        return out
    except Exception as e:                                 # noqa: BLE001
        warnings.warn(f"ES-RNN (ratio) failed: {e}")
        return (None, None) if return_state else None


# ------------------------------------------------------------ NeuralProphet
def _neuralprophet_pkg(MAT, weeks, tr_end, h, epochs):
    """The real ``neuralprophet`` package, global model over an ``ID`` column."""
    import pandas as pd
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from neuralprophet import NeuralProphet, set_log_level
    set_log_level("ERROR")
    S = MAT.shape[0]
    ds = pd.to_datetime(weeks[:tr_end])
    long = pd.concat(
        [pd.DataFrame({"ds": ds, "y": MAT[i, :tr_end].astype(float), "ID": str(i)})
         for i in range(S)], ignore_index=True)
    m = NeuralProphet(n_lags=min(4, tr_end - h), n_forecasts=h,
                      yearly_seasonality=False, weekly_seasonality=False,
                      daily_seasonality=False, growth="off", epochs=epochs,
                      learning_rate=1e-2, normalize="soft", drop_missing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(long, freq="W", minimal=True)
        fut = m.make_future_dataframe(long, periods=h, n_historic_predictions=False)
        fc = m.predict(fut)
    out = np.zeros((S, h), float)
    yh = [c for c in fc.columns if c.startswith("yhat")][:h]
    for i, g in fc.groupby("ID"):
        out[int(i)] = np.nan_to_num(np.clip([g.iloc[-1][c] for c in yh], 0, None))
    return out


def _neuralprophet_torch(MAT, tr_end, h, *, n_lags=4, period=4, n_fourier=2,
                         epochs=160, lr=6e-3, seed=0):
    """NeuralProphet's *method*, from scratch in torch, for when the package
    can't run: additive  y(t) = level + seasonality(t) + AR-Net(last n_lags).

      * level       — per-series offset, shrunk toward the series' robust median
                      (NeuralProphet ``growth='off'``: no slope / changepoints,
                      because 13 weekly points don't support a trend estimate)
      * seasonality — global Fourier terms on ``period`` weeks, scaled per series
      * AR-Net      — one shared linear map from the last n_lags residuals to the
                      next h steps, weight-decayed hard so it only nudges
    Fitted jointly by SGD on the training weeks only.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    y = np.clip(MAT[:, :tr_end].astype(np.float32), 0, None)
    S, T = y.shape
    if T < n_lags + h + 2 or S == 0:
        return None
    Y = torch.tensor(y)
    t_idx = torch.arange(T, dtype=torch.float32)
    scale = Y.mean(1, keepdim=True).clamp(min=1e-3)
    Yn = Y / scale
    # robust anchor that stays > 0 for intermittent series: mean of the nonzero
    # weeks, deflated by how often the SKU actually sells (a Croston-style rate)
    nz = Yn * (Yn > 0)
    cnt = (Yn > 0).sum(1).clamp(min=1)
    anchor = (nz.sum(1) / cnt) * ((Yn > 0).float().mean(1))       # (S,)
    anchor = torch.where(anchor > 0, anchor, Yn.mean(1))

    level = anchor.clone().detach().requires_grad_(True)          # (S,)
    fourier = torch.zeros(2 * n_fourier, requires_grad=True)
    seas_amp = torch.zeros(S, requires_grad=True)
    arnet = nn.Linear(n_lags, h)
    for p in arnet.parameters():                                  # start near zero
        nn.init.zeros_(p)

    def season(tt):
        ang = (2 * np.pi * (tt[:, None] + 1)
               * torch.arange(1, n_fourier + 1) / period)
        return torch.cat([torch.sin(ang), torch.cos(ang)], 1)     # (len(tt), 2F)

    S_basis = season(t_idx)
    params = [level, fourier, seas_amp] + list(arnet.parameters())
    opt = torch.optim.Adam([
        {"params": [level, fourier, seas_amp]},
        {"params": arnet.parameters(), "weight_decay": 4e-3},
    ], lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        se = seas_amp[:, None] * (S_basis @ fourier)[None, :]
        decomp = level[:, None] + se                              # (S, T)
        resid = Yn - decomp
        loss = 0.0
        for s0 in range(n_lags, T - h):
            pred = decomp[:, s0:s0 + h] + arnet(resid[:, s0 - n_lags:s0])
            loss = loss + (pred - Yn[:, s0:s0 + h]).abs().mean()
        loss = loss / max(T - h - n_lags, 1)
        loss = loss + 8e-3 * (level - anchor).pow(2).mean()           # shrink to the anchor
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

    with torch.no_grad():
        se_f = seas_amp[:, None] * (season(torch.arange(T, T + h, dtype=torch.float32))
                                    @ fourier)[None, :]
        resid = Yn - (level[:, None] + seas_amp[:, None] * (S_basis @ fourier)[None, :])
        ar_f = arnet(resid[:, T - n_lags:T])
        out = (level[:, None] + se_f + ar_f).clamp(min=0)
        # a still-selling SKU should not be forecast to zero — floor at its rate
        rate = anchor.clamp(min=0)[:, None]
        active = (Yn[:, -4:].sum(1, keepdim=True) > 0).float()
        out = torch.maximum(out, active * rate)
        out = (out * scale).cpu().numpy()
    return out.astype(float)


def neuralprophet_forecast(MAT: np.ndarray, weeks: list[str], tr_end: int, h: int,
                           *, epochs: int = 40):
    """Global NeuralProphet forecast: additive level + Fourier seasonality +
    AR-Net, fitted by SGD across every SKU on the training weeks only.

    The ``neuralprophet`` PyPI package (v0.9) does not run against the installed
    torch / Lightning, so this is a from-scratch torch implementation of the same
    decomposition. Set ``WMS_NEURALPROPHET_PKG=1`` to try the real package once
    its dependencies are pinned to compatible versions.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return None
    try:
        if MAT.shape[0] == 0 or MAT.shape[1] < h + 6:
            return None
        import os
        if os.environ.get("WMS_NEURALPROPHET_PKG") == "1":
            try:
                return _neuralprophet_pkg(MAT, weeks, tr_end, h, epochs)
            except Exception as e:                        # noqa: BLE001
                warnings.warn(f"neuralprophet package failed ({e}); using built-in")
        return _neuralprophet_torch(MAT, tr_end, h)
    except Exception as e:                                # noqa: BLE001
        warnings.warn(f"NeuralProphet failed: {e}")
        return None
