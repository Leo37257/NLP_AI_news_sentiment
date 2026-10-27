"""Join the Stage 3b BERTopic sectors to the Stage 5 sentence sentiment and summarise sentiment by sector and year.

Article score = mean over its AI-relevant sentences of P(positive) - P(negative), from the fine-tuned RoBERTa model.
Sector score = mean article score, with a 95% bootstrap CI over articles (so long articles do not dominate).
The 2,000 hand-labelled sentences are left out. Press-release wires are flagged and the sector ranking is recomputed
without them, because press releases are written to be positive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from map_industries import GENERIC_ORIGINAL, MAP, NOISE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sector_sentiment"
SEED, B = 42, 2000
UNASSIGNED = "Unassigned (outlier)"
PRESS_WIRES = ("einpresswire.com", "prnewswire.", "globenewswire.com", "businesswire.com", "newswire.ca", "prweb.com",
               "webnewswire.com", "sbwire.com", "emailwire.com", "accesswire.com", "newsfilecorp.com")
FULL_YEARS = (2023, 2024, 2025)  # 2022 starts thin and 2026 stops in February
MIN_ARTICLES = 100


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(B, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def summarise(art: pd.DataFrame, by: list[str], rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for key, g in art.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        lo, hi = boot_ci(g.net.to_numpy(), rng)
        rows.append({**dict(zip(by, key)), "articles": len(g), "sentences": int(g.n.sum()),
                     "net_sentiment": round(float(g.net.mean()), 4), "ci_low": lo, "ci_high": hi,
                     "share_negative_sentences": round(float(g.n_neg.sum() / g.n.sum()), 4),
                     "share_press_release": round(float(g.press.mean()), 4)})
    return pd.DataFrame(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    s = pd.read_parquet(ROOT / "data/sent_sentence.parquet",
                        columns=["url", "date", "sentence", "pred_label", "prob_negative", "prob_positive"])
    labelled = pd.read_csv(ROOT / "sent_annotation_labeled_2000.csv", usecols=["url", "sentence"])
    s = s.merge(labelled.assign(_l=1), on=["url", "sentence"], how="left")
    s = s[s._l.isna()].drop(columns=["_l", "sentence"])

    d = pd.read_parquet(ROOT / "results/bertopic_v2/doc_topics_with_original_labels.parquet",
                        columns=["url", "domain", "bertopic_topic_40", "industry"])
    d["sector"] = d.bertopic_topic_40.map(lambda t: MAP[t][0] if t in MAP else UNASSIGNED)
    d["press"] = d.domain.astype(str).apply(lambda x: any(w in x for w in PRESS_WIRES))

    j = s.merge(d[["url", "sector", "industry", "press"]], on="url", how="inner")
    j["neg"] = j.pred_label.eq("negative")
    art = j.groupby("url", sort=False).agg(
        sector=("sector", "first"), industry=("industry", "first"), press=("press", "first"),
        year=("date", lambda x: pd.Timestamp(x.iloc[0]).year),
        pos=("prob_positive", "mean"), negp=("prob_negative", "mean"), n=("pred_label", "size"), n_neg=("neg", "sum"))
    art["net"] = art.pos - art.negp

    OUT.mkdir(parents=True, exist_ok=True)
    content = art[~art.sector.isin([NOISE, UNASSIGNED])]
    sec = summarise(content, ["sector"], rng).sort_values("net_sentiment")
    no_press = summarise(content[~content.press], ["sector"], rng)[["sector", "net_sentiment", "articles"]]
    sec = sec.merge(no_press.rename(columns={"net_sentiment": "net_sentiment_no_press",
                                             "articles": "articles_no_press"}), on="sector")
    sec.to_csv(OUT / "sector_sentiment.csv", index=False)

    yr = summarise(content[content.year.isin(FULL_YEARS)], ["sector", "year"], rng)
    yr = yr.sort_values(["sector", "year"])
    yr.to_csv(OUT / "sector_sentiment_by_year.csv", index=False)
    first, last = FULL_YEARS[0], FULL_YEARS[-1]
    changes = []
    for sector, g in content.groupby("sector"):
        a, b = g[g.year == first].net.to_numpy(), g[g.year == last].net.to_numpy()
        if min(len(a), len(b)) < MIN_ARTICLES:
            continue
        boot = (b[rng.integers(0, len(b), size=(B, len(b)))].mean(axis=1)
                - a[rng.integers(0, len(a), size=(B, len(a)))].mean(axis=1))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        changes.append({"sector": sector, f"articles_{first}": len(a), f"articles_{last}": len(b),
                        "change": round(float(b.mean() - a.mean()), 4), "ci_low": round(float(lo), 4),
                        "ci_high": round(float(hi), 4), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    change = pd.DataFrame(changes).sort_values("change")
    change.to_csv(OUT / f"sector_change_{first}_{last}.csv", index=False)

    gen = summarise(art[art.industry == GENERIC_ORIGINAL], ["sector"], rng).sort_values("net_sentiment")
    gen.to_csv(OUT / "kmeans_generic_bucket_by_sector.csv", index=False)
    gen_all = art[art.industry == GENERIC_ORIGINAL]
    gen_big = gen[(gen.articles >= MIN_ARTICLES) & ~gen.sector.isin([NOISE, UNASSIGNED])]

    rank_corr = float(sec[["net_sentiment", "net_sentiment_no_press"]].rank().corr().iloc[0, 1])
    press_corr = float(sec[["net_sentiment", "share_press_release"]].corr(method="spearman").iloc[0, 1])
    summary = {
        "sentences_used": int(len(j)), "articles": int(len(art)), "articles_with_sector": int(len(content)),
        "overall_net_sentiment": round(float(content.net.mean()), 4),
        "most_negative": sec.head(4)[["sector", "net_sentiment", "ci_low", "ci_high", "articles"]].to_dict("records"),
        "most_positive": sec.tail(4)[["sector", "net_sentiment", "ci_low", "ci_high", "articles"]].to_dict("records"),
        "press_release": {"share_of_articles": round(float(content.press.mean()), 4),
                          "spearman_sector_net_vs_press_share": round(press_corr, 3),
                          "spearman_rank_with_vs_without_press": round(rank_corr, 3)},
        f"change_{FULL_YEARS[0]}_to_{FULL_YEARS[-1]}_ci_excludes_zero": change[change.ci_excludes_zero][
            ["sector", "change", "ci_low", "ci_high"]].to_dict("records"),
        "kmeans_generic_bucket": {
            "articles": int(len(gen_all)), "net_sentiment": round(float(gen_all.net.mean()), 4),
            "sector_range_within_bucket": [gen_big.iloc[0][["sector", "net_sentiment"]].to_dict(),
                                           gen_big.iloc[-1][["sector", "net_sentiment"]].to_dict()]},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))
    print(sec.to_string(index=False))


if __name__ == "__main__":
    main()
