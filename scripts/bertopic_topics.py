"""Upgrade of stage 3: BERTopic (sentence embeddings -> UMAP -> HDBSCAN -> c-TF-IDF) on the cleaned AI-news corpus,
compared with the original TF-IDF + K-Means topics on the same coherence and diversity measures.

    python scripts/bertopic_topics.py --source clean --out results/bertopic                          # first run
    python scripts/bertopic_topics.py --source hybrid --reduce-to 40,14 --out results/bertopic_v2   # final run
    python scripts/bertopic_topics.py --source sentences --sample 3000      # quick smoke test without the full corpus

Outputs: topic_info.csv, doc_topics_bertopic.parquet, metrics.json (and a cached embeddings .npy).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
# identical to notebook 03, so both topic models remove the same 346 terms
CUSTOM_STOPWORDS = {"ai", "artificial", "intelligence", "technology", "tech", "said", "new", "use", "using", "time", "year",
                    "people", "like", "just", "make", "help", "real", "based", "information", "content", "world", "news",
                    "com", "2023", "2024", "2025", "article", "articles"}
STOPWORDS = sorted(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))
SEED = 42
# listing / storefront sites whose pages are templates, not news (crypto price pages, trading-robot shop,
# AI-wallpaper gallery, course catalogue, job board); found because v1 clustered them into their own topics
LISTING_DOMAINS = ("mexc.", "mql5.com", "nativestew.com", "coursera.org", "vakanser.se")
TITLE_PREFIX = r"^\s*Tolerance\.ca®\s*-\s*"
TITLE_SUFFIX = r"\s+[-|–—]\s+[^-|–—]{2,45}$"          # " - CNA", " | Daily Mail Online", " - Stock market news | ..."


def clean_title(t: str) -> str:
    import re
    t = re.sub(TITLE_PREFIX, "", str(t or ""))
    for _ in range(2):
        t = re.sub(TITLE_SUFFIX, "", t)
    return t.strip()


def load_corpus(args) -> pd.DataFrame:
    if args.source == "sentences":       # fallback: rebuild each article from its AI-relevant sentences
        s = pd.read_parquet(args.sentences, columns=["url", "date", "sentence"])
        df = s.groupby("url", sort=False).agg(date=("date", "first"), text=("sentence", " ".join)).reset_index()
        df["title"] = ""
    elif args.source == "hybrid":
        meta = pd.read_parquet(args.input, columns=["url", "date", "domain", "title"])
        meta = meta[~meta["domain"].astype(str).str.contains("|".join(LISTING_DOMAINS), regex=True)]
        s = pd.read_parquet(args.sentences, columns=["url", "sentence"])
        sent = s.groupby("url", sort=False)["sentence"].apply(" ".join).rename("text")
        df = meta.merge(sent, left_on="url", right_index=True, how="inner")
        df["title"] = df["title"].map(clean_title)
        df = df.reset_index(drop=True)
    else:
        import pyarrow.parquet as pq
        names = pq.ParquetFile(args.input).schema_arrow.names
        text_col = args.text_col if args.text_col in names else next(
            c for c in ("text_clean", "clean_text", "text", "content", "body") if c in names)
        df = pd.read_parquet(args.input, columns=[c for c in ("url", "date", "domain", "title", text_col) if c in names])
        df = df.rename(columns={text_col: "text"})
        if "title" not in df.columns:
            df["title"] = ""
    df = df[df["text"].astype(str).str.len() > 0].reset_index(drop=True)
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def embed(docs: list[str], cache: Path, model_name: str) -> np.ndarray:
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(docs):
            return emb
    import torch
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(docs, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def tokenize(docs: list[str]) -> list[list[str]]:
    analyzer = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1, 1)).build_analyzer()
    return [analyzer(d) for d in docs]


def coherence_and_diversity(topics: list[list[str]], tokenized: list[list[str]]) -> dict:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    dictionary = Dictionary(tokenized)
    words = [[w for w in t if w in dictionary.token2id][:10] for t in topics]
    words = [w for w in words if len(w) >= 3]
    out = {"n_topics_scored": len(words),
           "diversity": round(len({w for t in words for w in t}) / max(sum(len(t) for t in words), 1), 4)}
    for measure in ("c_npmi", "c_v"):
        cm = CoherenceModel(topics=words, texts=tokenized, dictionary=dictionary, coherence=measure, processes=1)
        out[measure] = round(float(cm.get_coherence()), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["clean", "sentences", "hybrid"], default="clean",
                   help="hybrid = cleaned title + the article's AI-relevant sentences from stage 5")
    p.add_argument("--input", default=str(ROOT / "data/clean_news_final_project.parquet"))
    p.add_argument("--sentences", default=str(ROOT / "data/sent_sentence.parquet"))
    p.add_argument("--text-col", default="text_clean")
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--max-chars", type=int, default=2000, help="characters per article fed to the embedder")
    p.add_argument("--min-cluster-size", type=int, default=150)
    p.add_argument("--embedder", default="all-MiniLM-L6-v2")
    p.add_argument("--kmeans-topics", default=str(ROOT / "topic_summary.csv"))
    p.add_argument("--reduce-to", default="", help="comma-separated topic counts to merge down to, e.g. 40,14")
    p.add_argument("--out", default=str(ROOT / "results/bertopic"))
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = load_corpus(args)
    docs = (df["title"].fillna("") + ". " + df["text"].astype(str).str.slice(0, args.max_chars)).str.strip(". ").tolist()
    print(f"documents: {len(docs):,}")
    emb = embed(docs, out / f"embeddings_{len(docs)}.npy", args.embedder)
    print(f"embeddings: {emb.shape} in {time.time() - t0:.0f}s")

    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, min_samples=10, metric="euclidean",
                  cluster_selection_method="eom", prediction_data=True)
    # c-TF-IDF runs on one concatenated document per topic, so document-frequency cut-offs must stay loose
    vec = CountVectorizer(stop_words=STOPWORDS, ngram_range=(1, 2), min_df=1)
    model = BERTopic(embedding_model=None, umap_model=umap_model, hdbscan_model=hdb, vectorizer_model=vec,
                     top_n_words=10, calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(docs, embeddings=emb)
    raw_outliers = float(np.mean(np.array(topics) == -1))
    topics = model.reduce_outliers(docs, topics, strategy="c-tf-idf", threshold=0.1)
    model.update_topics(docs, topics=topics, vectorizer_model=vec)
    print(f"BERTopic fitted in {time.time() - t0:.0f}s")

    info = model.get_topic_info()
    info["top_words"] = info["Topic"].map(lambda t: ", ".join(w for w, _ in (model.get_topic(t) or [])[:10]))
    info[["Topic", "Count", "Name", "top_words"]].to_csv(out / "topic_info.csv", index=False)
    keep = [c for c in ("url", "date", "domain", "title") if c in df.columns]
    df[keep].assign(bertopic_topic=topics).to_parquet(out / "doc_topics_bertopic.parquet", index=False)

    sample_idx = np.random.default_rng(SEED).choice(len(docs), size=min(20000, len(docs)), replace=False)
    tokenized = tokenize([docs[i] for i in sample_idx])
    bt_words = [[w for w, _ in model.get_topic(t)] for t in sorted(set(topics)) if t != -1]
    bt_words = [[w for x in t for w in x.split()][:10] for t in bt_words]      # bigrams -> unigrams for scoring
    metrics = {"documents": len(docs), "source": args.source, "embedder": args.embedder,
               "min_cluster_size": args.min_cluster_size,
               "bertopic": {"n_topics": int(len(set(topics)) - (1 if -1 in topics else 0)),
                            "outlier_share_before_reduction": round(raw_outliers, 4),
                            "outlier_share_after_reduction": round(float(np.mean(np.array(topics) == -1)), 4),
                            **coherence_and_diversity(bt_words, tokenized)}}
    km = pd.read_csv(args.kmeans_topics)
    km_kept = km[km["topic_keep"].astype(str).str.lower().isin(["true", "1", "yes"])] if "topic_keep" in km else km
    km_words = [[w.strip() for w in s.split(",")] for s in km_kept["keywords"]]
    metrics["kmeans_baseline"] = {"n_topics": len(km_words), **coherence_and_diversity(km_words, tokenized)}
    doc_out = df[keep].assign(bertopic_topic=topics)
    for n in [int(x) for x in args.reduce_to.split(",") if x.strip()]:
        model.reduce_topics(docs, nr_topics=n + 1)        # +1 because the outlier topic -1 is counted
        t_n = model.topics_
        words_n = [[w for x in [w for w, _ in model.get_topic(t)] for w in x.split()][:10]
                   for t in sorted(set(t_n)) if t != -1]
        info_n = model.get_topic_info()
        info_n["top_words"] = info_n["Topic"].map(lambda t: ", ".join(w for w, _ in (model.get_topic(t) or [])[:10]))
        info_n[["Topic", "Count", "Name", "top_words"]].to_csv(out / f"topic_info_{n}.csv", index=False)
        doc_out[f"bertopic_topic_{n}"] = t_n
        metrics[f"bertopic_reduced_to_{n}"] = {"n_topics": int(len(set(t_n)) - (1 if -1 in t_n else 0)),
                                              "outlier_share": round(float(np.mean(np.array(t_n) == -1)), 4),
                                              **coherence_and_diversity(words_n, tokenized)}
        print(f"reduced to {n}: {metrics[f'bertopic_reduced_to_{n}']}")
    doc_out.to_parquet(out / "doc_topics_bertopic.parquet", index=False)
    metrics["runtime_seconds"] = round(time.time() - t0)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
