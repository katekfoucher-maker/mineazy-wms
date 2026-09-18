"""Rich / questionary presentation helpers."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(width=None if sys.stdout.isatty() else 150)


def banner(title: str, subtitle: str = "") -> None:
    console.print(Panel.fit(f"[bold]{title}[/bold]" + (f"\n{subtitle}" if subtitle else ""),
                            border_style="cyan"))


def menu(title: str, choices: Sequence[str]) -> Optional[str]:
    try:
        return questionary.select(title, choices=list(choices), qmark="»").ask()
    except (KeyboardInterrupt, EOFError):
        return None


def ask_text(msg: str, default: str = "") -> Optional[str]:
    try:
        return questionary.text(msg, default=default).ask()
    except (KeyboardInterrupt, EOFError):
        return None


def ask_int(msg: str, default: Optional[int] = None, minimum: Optional[int] = None) -> Optional[int]:
    while True:
        raw = ask_text(msg + (f" [{default}]" if default is not None else ""))
        if raw is None:
            return None
        raw = raw.strip()
        if raw == "" and default is not None:
            return default
        try:
            v = int(raw)
        except ValueError:
            console.print("[red]enter a whole number[/red]")
            continue
        if minimum is not None and v < minimum:
            console.print(f"[red]must be >= {minimum}[/red]")
            continue
        return v


def confirm(msg: str, default: bool = False) -> bool:
    try:
        return bool(questionary.confirm(msg, default=default).ask())
    except (KeyboardInterrupt, EOFError):
        return False


def pause() -> None:
    try:
        questionary.text("press Enter to continue").ask()
    except (KeyboardInterrupt, EOFError):
        pass


def show_df(df: pd.DataFrame, title: str = "", max_rows: int = 40) -> None:
    if df is None or len(df) == 0:
        console.print(f"[yellow]{title or 'result'}: no rows[/yellow]")
        return
    t = Table(title=title or None, header_style="bold white on grey23")
    for col in df.columns:
        t.add_column(str(col), overflow="fold")
    for _, row in df.head(max_rows).iterrows():
        t.add_row(*[_fmt(v) for v in row.tolist()])
    console.print(t)
    if len(df) > max_rows:
        console.print(f"[dim]... {len(df) - max_rows} more rows[/dim]")


def show_kpis(d: dict, title: str = "KPIs") -> None:
    t = Table(title=title, show_header=False, border_style="grey42")
    t.add_column("metric", style="bold")
    t.add_column("value", justify="right")
    for k, v in d.items():
        t.add_row(str(k), _fmt(v))
    console.print(t)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if isinstance(v, float):
        return f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def open_file(path: Path | str) -> None:
    path = str(path)
    console.print(f"[green]saved[/green] {path}")
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa: S606
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}" >/dev/null 2>&1 &')
    except Exception:
        pass
