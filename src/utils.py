from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def ensure_directories(paths: Iterable[str | Path]) -> None:
    """Create output directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, output_path: str | Path, dpi: int = 130) -> None:
    """Save a matplotlib figure and close it to avoid notebook memory issues."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_text_report(content: str, output_path: str | Path) -> None:
    """Write a plain text report to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
