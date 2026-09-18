"""ES-RNN weekly forecaster (torch). Skipped when torch is unavailable."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from wms.analytics.weekly_neural import (                # noqa: E402
    esrnn_forecast, esrnn_ratio_forecast, neuralprophet_forecast,
)


def _panel(n_series=40, n_weeks=17, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        base = rng.uniform(2, 40)
        drift = rng.uniform(-0.5, 0.5)
        y = np.maximum(0, base + drift * np.arange(n_weeks)
                       + rng.normal(0, base * 0.3, n_weeks)).round()
        if i % 4 == 0:                                    # some intermittent series
            y[rng.random(n_weeks) < 0.5] = 0
        rows.append(y)
    return np.array(rows, dtype=np.float32)


def test_esrnn_is_a_genuine_multistep_forecast_from_train_only():
    MAT = _panel()
    S, W = MAT.shape
    h = 4
    tr_end = W - h
    F = esrnn_forecast(MAT, tr_end, h, epochs=15)
    assert F is not None
    assert F.shape == (S, h)
    assert np.isfinite(F).all() and (F >= 0).all()

    # forecast depends only on the training weeks: scrambling the held-out
    # columns must not change the output at all
    MAT2 = MAT.copy()
    MAT2[:, tr_end:] = 999.0
    F2 = esrnn_forecast(MAT2, tr_end, h, epochs=15)
    assert np.allclose(F, F2)

    # it is not a flat repeat — at least some series move across the horizon
    moves = (F.std(axis=1) > 1e-6).mean()
    assert moves > 0.5


def test_esrnn_returns_none_when_history_too_short():
    MAT = _panel(n_series=10, n_weeks=8)
    assert esrnn_forecast(MAT, tr_end=4, h=4) is None


def test_esrnn_ratio_is_a_bounded_multiplier_on_the_level():
    MAT = _panel()
    S, W = MAT.shape
    h = 1
    tr_end = W - h
    F = esrnn_ratio_forecast(MAT, tr_end, h, epochs=20)
    assert F is not None and F.shape == (S, h)
    assert np.isfinite(F).all() and (F >= 0).all()

    # forecast is base * r with r in [lo, hi]: recover the implied multiplier
    # against the same base the model uses (recent level floored at the rate)
    import inspect
    from wms.analytics.weekly_neural import _croston_rate
    sig = inspect.signature(esrnn_ratio_forecast).parameters
    lo, hi = sig["lo"].default, sig["hi"].default
    tr = MAT[:, :tr_end]
    rate = np.array([_croston_rate(tr[i]) for i in range(S)])
    recent = tr[:, -min(4, tr_end):].mean(1)
    dw = min(6, tr_end)
    dead = ~tr[:, -dw:].any(axis=1) if dw else np.zeros(S, bool)
    base = np.where(dead, recent, np.maximum(recent, rate))
    active = base > 1e-6
    r = F[active, 0] / base[active]
    assert (r >= lo - 1e-3).all() and (r <= hi + 1e-3).all()

    # trained on the training weeks only — held-out columns must not matter
    MAT2 = MAT.copy()
    MAT2[:, tr_end:] = 999.0
    assert np.allclose(F, esrnn_ratio_forecast(MAT2, tr_end, h, epochs=20))


def test_esrnn_ratio_checkpoint_reuse():
    MAT = _panel()
    S, W = MAT.shape
    tr_end = W - 1

    F0, state = esrnn_ratio_forecast(MAT, tr_end, 1, epochs=25, return_state=True)
    assert F0 is not None and state is not None
    assert set(state) == {"hp", "net_state"}
    assert state["hp"] == {"hidden": 16, "win": 4, "h": 1, "n_feats": 5}

    # reusing the checkpoint: loads the frozen net, only re-fits alpha -> still a
    # valid bounded forecast, and it does NOT peek at the held-out weeks
    F1 = esrnn_ratio_forecast(MAT, tr_end, 1, checkpoint=state, epochs=25)
    assert F1.shape == (S, 1) and np.isfinite(F1).all() and (F1 >= 0).all()
    MAT2 = MAT.copy(); MAT2[:, tr_end:] = 999.0
    assert np.allclose(F1, esrnn_ratio_forecast(MAT2, tr_end, 1, checkpoint=state,
                                                epochs=25))

    # a checkpoint whose hyper-params don't match is ignored (fresh train, no crash)
    bad = {"hp": {"hidden": 99, "win": 4, "h": 1, "n_feats": 5},
           "net_state": state["net_state"]}
    assert esrnn_ratio_forecast(MAT, tr_end, 1, checkpoint=bad, epochs=10).shape == (S, 1)


def test_neuralprophet_forecast_from_train_only():
    MAT = _panel()
    S, W = MAT.shape
    h = 4
    tr_end = W - h
    F = neuralprophet_forecast(MAT, [f"2026-{1 + i // 4:02d}-{1 + 7 * (i % 4):02d}"
                                     for i in range(W)], tr_end, h)
    assert F is not None
    assert F.shape == (S, h)
    assert np.isfinite(F).all() and (F >= 0).all()

    MAT2 = MAT.copy()
    MAT2[:, tr_end:] = 999.0
    F2 = neuralprophet_forecast(MAT2, [""] * W, tr_end, h)
    assert np.allclose(F, F2)                             # no leakage from held-out weeks
