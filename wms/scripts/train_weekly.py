"""Offline trainer for the weekly ``esrnn_ratio`` forecast network.

    python -m wms.scripts.train_weekly              # thorough: ~400 epochs + val
    python -m wms.scripts.train_weekly --quick      # ~120 epochs, for a fast pass
    python -m wms.scripts.train_weekly --epochs 800

Reads every weekly sales file in ``data/weekly_sales/``, fits the shared LSTM +
head PROPERLY (many epochs, a validation tail with early stopping), and writes:

    output/weekly_esrnn_ratio.pt      the trained network weights
    output/weekly_esrnn_ratio.json    when / on what it was trained + hold-out score

The running app auto-loads that network on its next request: instead of training
from scratch every restart (~minutes), it loads these weights, freezes them, and
only re-fits each series' smoothing alpha (seconds) — and still adapts when a
branch or SKU is added later. Delete the two files to go back to on-demand
training.
"""
from __future__ import annotations

import argparse
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="train_weekly", description=__doc__)
    ap.add_argument("--epochs", type=int, default=400,
                    help="max training epochs (default 400; early-stops on val)")
    ap.add_argument("--val-weeks", type=int, default=3,
                    help="most-recent training windows held out for early stopping")
    ap.add_argument("--quick", action="store_true",
                    help="fast pass (~120 epochs) instead of the full fit")
    a = ap.parse_args(argv)

    from wms.analytics import weekly_forecast as wf

    t0 = time.time()
    print(f"training esrnn_ratio  (epochs<= {120 if a.quick else a.epochs}, "
          f"val_weeks={a.val_weeks}, quick={a.quick}) ...", flush=True)
    try:
        meta = wf.train_and_save(epochs=a.epochs, val_weeks=a.val_weeks,
                                 quick=a.quick)
    except Exception as e:                                # noqa: BLE001
        print(f"FAILED: {e}", file=sys.stderr)
        return 1

    dt = time.time() - t0
    print(f"\nsaved -> {wf._ratio_ckpt_path()}")
    if wf._gbm_ckpt_path().exists():
        print(f"saved -> {wf._gbm_ckpt_path()}")
    print(f"  trained_at   : {meta['trained_at']}   ({dt:.0f}s)")
    print(f"  data         : {meta['n_series']} series x {meta['n_weeks']} weeks "
          f"({', '.join(meta['branches'])})")
    print(f"  epochs        : {meta['epochs']}  (val_weeks {meta['val_weeks']})")
    ms = meta.get("model_scores") or {}
    if ms:
        n = meta.get("rolling_origins", "?")
        print(f"\n  rolling-origin CV ({n} origins, in-stock, what really sold):")
        for name in ("blend_learned", "blend_fixed", "esrnn_ratio", "gbm",
                     "lgbm", "lstm"):
            if name in ms and ms[name]:
                print(f"    {name:14} WAPE {ms[name]['wape']:5.1f}   "
                      f"bias {ms[name]['bias']:+.1f}")
    else:
        print(f"  hold-out : WAPE {meta['holdout_wape']}  "
              f"bias {meta['holdout_bias']:+}")
    bw = meta.get("blend_weights") or {}
    if bw:
        print(f"\n  blend ({meta.get('blend_mode', '?')}): "
              + ", ".join(f"{k} {v:.2f}" for k, v in bw.items()))
    print(f"  serving model : {wf.forced_model() or 'auto (bias-aware pick)'}")
    print("the app will pick this up on its next request.")
    return 0


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
