"""Shared helpers for report figures and diagrams that belong in LaTeX assets."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
LATEX_ASSETS_DIR = ROOT / "latex" / "assets"
LEGACY_REPORT_DIR = ROOT / "data" / "report"
LEGACY_REPORT_EXT_DIR = ROOT / "data" / "report_ext"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def paths_match(lhs: str | Path, rhs: str | Path) -> bool:
    return Path(lhs).resolve(strict=False) == Path(rhs).resolve(strict=False)


def save_figure(
    fig: plt.Figure,
    out_path: str | Path,
    *,
    mirror_paths: Iterable[str | Path] = (),
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> str:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)

    for mirror_path in mirror_paths:
        mirror_path = Path(mirror_path)
        if paths_match(mirror_path, out_path):
            continue
        ensure_dir(mirror_path.parent)
        fig.savefig(mirror_path, dpi=dpi, bbox_inches=bbox_inches)

    plt.close(fig)
    return str(out_path)


def write_text_asset(
    content: str,
    out_path: str | Path,
    *,
    mirror_paths: Iterable[str | Path] = (),
    encoding: str = "utf-8",
) -> str:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(content, encoding=encoding)

    for mirror_path in mirror_paths:
        mirror_path = Path(mirror_path)
        if paths_match(mirror_path, out_path):
            continue
        ensure_dir(mirror_path.parent)
        mirror_path.write_text(content, encoding=encoding)

    return str(out_path)


def copy_asset(
    source_path: str | Path,
    out_path: str | Path,
    *,
    mirror_paths: Iterable[str | Path] = (),
) -> str:
    source_path = Path(source_path)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    if not paths_match(source_path, out_path):
        shutil.copy2(source_path, out_path)

    for mirror_path in mirror_paths:
        mirror_path = Path(mirror_path)
        if paths_match(mirror_path, out_path):
            continue
        ensure_dir(mirror_path.parent)
        shutil.copy2(source_path, mirror_path)

    return str(out_path)
