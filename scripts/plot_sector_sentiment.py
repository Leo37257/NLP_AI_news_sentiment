"""Draw docs/sector_sentiment.png from results/sector_sentiment/sector_sentiment.csv (run scripts/sector_sentiment.py first)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEG, POS = "#e34948", "#2a78d6"  # diverging poles; position already shows the sign, colour repeats it


def main() -> None:
    sec = pd.read_csv(ROOT / "results/sector_sentiment/sector_sentiment.csv").sort_values("net_sentiment")
    overall = json.loads((ROOT / "results/sector_sentiment/summary.json").read_text())["overall_net_sentiment"]
    fig, ax = plt.subplots(figsize=(9, 8.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    y = range(len(sec))
    colors = [NEG if v < 0 else POS for v in sec.net_sentiment]
    ax.hlines(y, sec.ci_low, sec.ci_high, color=colors, linewidth=2, zorder=2)
    ax.scatter(sec.net_sentiment, y, s=46, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.axvline(0, color=INK_2, linewidth=1, zorder=1)
    ax.axvline(overall, color=INK_2, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    ax.text(overall + 0.015, len(sec) - 0.4, f"all sectors {overall:+.2f}", color=INK_2, fontsize=8.5, va="center")
    labels = [f"{s}  ({n:,})" for s, n in zip(sec.sector, sec.articles)]
    ax.set_yticks(list(y), labels, fontsize=8.5, color=INK)
    for i, (v, hi, lo) in enumerate(zip(sec.net_sentiment, sec.ci_high, sec.ci_low)):
        if i < 4 or i >= len(sec) - 3:  # label only the extremes
            ax.text((hi + 0.02) if v >= 0 else (lo - 0.02), i, f"{v:+.2f}", fontsize=8, color=INK_2, va="center",
                    ha="left" if v >= 0 else "right")
    ax.set_xlim(-0.65, 1.0)
    ax.set_ylim(-0.8, len(sec) - 0.2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", colors=INK_2, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Net sentiment per article: mean P(positive) − P(negative), with 95% bootstrap CI", color=INK_2,
                  fontsize=8.5)
    ax.set_title("How AI news talks about each sector", loc="left", fontsize=12, color=INK, fontweight="bold", pad=22)
    ax.text(0, 1.012, "26 BERTopic sectors, 63,772 articles (article count in brackets). Fine-tuned RoBERTa, 2022–2026.",
            transform=ax.transAxes, fontsize=8.5, color=INK_2)
    fig.tight_layout()
    out = ROOT / "docs/sector_sentiment.png"
    fig.savefig(out, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
