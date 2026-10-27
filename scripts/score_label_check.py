"""Score the filled-in blind labeling sheet from scripts/make_label_check.py.

Per model: share of articles whose label was judged correct & specific (2), correct (2 or 1) and wrong (0), with
paired bootstrap 95% CIs, plus the head-to-head preference with an exact sign test. Writes aggregates only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "data/labeling/industry_label_check.xlsx"
KEY = ROOT / "data/labeling/label_check_key.csv"
OUT = ROOT / "results/label_check"
SEED, B = 42, 10_000


def ci(x: np.ndarray, rng: np.random.Generator) -> list[float]:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    return [round(float(v), 4) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="human", help="who filled in the sheet, recorded in the summary")
    args, _ = ap.parse_known_args()
    sheet = pd.read_excel(SHEET, sheet_name="Label")
    d = sheet.merge(pd.read_csv(KEY), on="id")
    d = d.dropna(subset=["A_judgment", "B_judgment", "better"])
    for col in ("A_judgment", "B_judgment"):
        d[col] = d[col].astype(str).str[0].astype(int)
    a_bt = d.a_is.eq("bertopic")
    d["score_bertopic"] = np.where(a_bt, d.A_judgment, d.B_judgment)
    d["score_kmeans"] = np.where(a_bt, d.B_judgment, d.A_judgment)
    better = d.better.astype(str).str.strip()
    d["winner"] = np.select([better.eq("tie"), better.eq("A") == a_bt], ["tie", "bertopic"], "kmeans")

    rng = np.random.default_rng(SEED)
    out = {"judge": args.judge, "articles_scored": int(len(d)), "articles_in_sheet": int(len(sheet))}
    for m in ("bertopic", "kmeans"):
        s = d[f"score_{m}"].to_numpy()
        out[m] = {"correct_and_specific": round(float((s == 2).mean()), 4), "ci_specific": ci((s == 2).astype(float), rng),
                  "correct": round(float((s >= 1).mean()), 4), "ci_correct": ci((s >= 1).astype(float), rng),
                  "wrong": round(float((s == 0).mean()), 4)}
    diff = (d.score_bertopic == 2).astype(float).to_numpy() - (d.score_kmeans == 2).astype(float).to_numpy()
    out["specific_difference_bertopic_minus_kmeans"] = {"value": round(float(diff.mean()), 4), "ci": ci(diff, rng)}
    wins, losses = int((d.winner == "bertopic").sum()), int((d.winner == "kmeans").sum())
    out["preference"] = {"bertopic": wins, "kmeans": losses, "tie": int((d.winner == "tie").sum()),
                         "sign_test_p": float(f"{binomtest(wins, wins + losses).pvalue:.2e}") if wins + losses else None}
    g = d[d.kmeans_industry == "Software / AI Platforms"]
    out["kmeans_generic_bucket"] = {"articles": int(len(g)),
                                    "bertopic_specific": round(float((g.score_bertopic == 2).mean()), 4) if len(g) else None,
                                    "kmeans_specific": round(float((g.score_kmeans == 2).mean()), 4) if len(g) else None}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
