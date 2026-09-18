"""CSV export helpers."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from wms.config import get_settings

settings = get_settings()


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_csv(rows: Iterable[dict], name: str, *, out_dir: Path | None = None) -> Path:
    rows = list(rows)
    out = Path(out_dir or settings.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{_ts()}.csv"
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def dataframe_to_csv(df, name: str, *, out_dir: Path | None = None) -> Path:
    out = Path(out_dir or settings.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}_{_ts()}.csv"
    df.to_csv(path, index=False)
    return path
