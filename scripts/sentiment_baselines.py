"""Put the Stage 5 RoBERTa sentiment score in context: baselines, per-class scores and bootstrap 95% CIs.

The split is rebuilt exactly as in notebook 05 (70/15/15, stratified, seed 42). The fine-tuned weights were not kept,
so RoBERTa's test predictions are read from the saved inference output (sent_sentence.parquet); they reproduce the
reported 82.3% accuracy and 0.764 macro F1. Baselines on the same 300 test sentences:
  - majority class
  - TF-IDF + logistic regression (C and class weights picked on the validation split)
  - an off-the-shelf 3-class sentiment model with no fine-tuning (cardiffnlp twitter-roberta, neutral -> uncertain_mixed)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/sentiment_eval"
LABELS = ["negative", "uncertain_mixed", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
SEED, B = 42, 10_000
OFF_THE_SHELF = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a = pd.read_csv(ROOT / "sent_annotation_labeled_2000.csv")
    a["y"] = a.label.map(L2I)
    tr, tmp = train_test_split(a, test_size=0.30, stratify=a.y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp.y, random_state=SEED)
    return tr, va, te


def tfidf_logreg(tr: pd.DataFrame, va: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, dict]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    xtr, xva, xte = vec.fit_transform(tr.sentence), vec.transform(va.sentence), vec.transform(te.sentence)
    best = None
    for c in (0.3, 1, 3, 10, 30):
        for cw in (None, "balanced"):
            m = LogisticRegression(C=c, class_weight=cw, max_iter=2000).fit(xtr, tr.y)
            f = f1_score(va.y, m.predict(xva), average="macro")
            if best is None or f > best[0]:
                best = (f, c, cw, m)
    return best[3].predict(xte), {"C": best[1], "class_weight": best[2], "val_macro_f1": round(best[0], 4)}


def off_the_shelf(te: pd.DataFrame) -> np.ndarray:
    import torch
    from transformers import pipeline
    device = "mps" if torch.backends.mps.is_available() else -1
    clf = pipeline("text-classification", model=OFF_THE_SHELF, device=device, truncation=True, max_length=128)
    to_ours = {"negative": "negative", "neutral": "uncertain_mixed", "positive": "positive"}
    preds = clf(te.sentence.tolist(), batch_size=32)
    return np.array([L2I[to_ours[p["label"].lower()]] for p in preds])


def scores(y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> dict:
    idx = rng.integers(0, len(y), size=(B, len(y)))
    acc = (y[idx] == p[idx]).mean(axis=1)
    f1 = np.array([f1_score(y[i], p[i], average="macro", labels=[0, 1, 2], zero_division=0) for i in idx[:2000]])
    return {"accuracy": round(accuracy_score(y, p), 4),
            "accuracy_ci": [round(float(v), 4) for v in np.percentile(acc, [2.5, 97.5])],
            "macro_f1": round(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0), 4),
            "macro_f1_ci": [round(float(v), 4) for v in np.percentile(f1, [2.5, 97.5])]}


def main() -> None:
    tr, va, te = split()
    s = pd.read_parquet(ROOT / "data/sent_sentence.parquet", columns=["url", "sentence", "pred_label"])
    m = te.merge(s.drop_duplicates(["url", "sentence"]), on=["url", "sentence"], how="left")
    assert m.pred_label.notna().all(), "every test sentence must have a saved prediction"
    y = te.y.to_numpy()
    preds = {"roberta_finetuned": m.pred_label.map(L2I).to_numpy(),
             "majority_class": np.full(len(y), L2I[tr.label.mode()[0]])}
    preds["tfidf_logreg"], lr_cfg = tfidf_logreg(tr, va, te)
    preds["off_the_shelf_sentiment"] = off_the_shelf(te)

    rng = np.random.default_rng(SEED)
    out = {"test_sentences": int(len(y)), "train_sentences": int(len(tr)), "val_sentences": int(len(va)),
           "test_class_counts": {l: int((y == i).sum()) for i, l in enumerate(LABELS)},
           "models": {k: scores(y, p, rng) for k, p in preds.items()}, "tfidf_logreg_config": lr_cfg,
           "off_the_shelf_model": OFF_THE_SHELF}
    r = preds["roberta_finetuned"]
    for base in ("tfidf_logreg", "off_the_shelf_sentiment"):
        b = preds[base]
        idx = rng.integers(0, len(y), size=(2000, len(y)))
        diff = np.array([f1_score(y[i], r[i], average="macro") - f1_score(y[i], b[i], average="macro") for i in idx])
        out[f"macro_f1_gain_vs_{base}"] = {"value": round(float(f1_score(y, r, average="macro") - f1_score(y, b, average="macro")), 4),
                                            "ci": [round(float(v), 4) for v in np.percentile(diff, [2.5, 97.5])]}
        acc_diff = ((y[idx] == r[idx]).mean(axis=1) - (y[idx] == b[idx]).mean(axis=1))
        out[f"accuracy_gain_vs_{base}"] = {"value": round(float((y == r).mean() - (y == b).mean()), 4),
                                            "ci": [round(float(v), 4) for v in np.percentile(acc_diff, [2.5, 97.5])]}

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, p in preds.items():
        pr, rc, f, n = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
        rows += [{"model": k, "class": LABELS[i], "precision": round(pr[i], 4), "recall": round(rc[i], 4),
                  "f1": round(f[i], 4), "support": int(n[i])} for i in range(3)]
    pd.DataFrame(rows).to_csv(OUT / "per_class.csv", index=False)
    pd.DataFrame(confusion_matrix(y, r, labels=[0, 1, 2]), index=[f"true_{l}" for l in LABELS],
                 columns=[f"pred_{l}" for l in LABELS]).to_csv(OUT / "roberta_confusion_matrix.csv")
    (OUT / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
