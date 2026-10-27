# AI News Sentiment Analysis: An End-to-End NLP Pipeline

## Overview

This project implements a complete natural language processing pipeline for analyzing a large corpus of AI-related news articles (approximately 200,000 documents, spanning 2022–2026). The goal is to identify which industries and companies are most frequently discussed in the context of artificial intelligence, characterize the nature of AI-related impacts (positive, negative, or uncertain), and surface the technologies and mechanisms driving AI adoption across sectors.

The pipeline is organized into six sequential stages, each implemented in a separate notebook. Intermediate outputs are saved as Parquet or CSV files and consumed by downstream stages.

---

## Research Questions

1. Which industries and companies are most likely to be impacted by AI over the next several years?
2. How are those industries and companies expected to be impacted — positively, negatively, or ambiguously — and through what mechanisms (e.g., automation, augmentation, generative AI)?
3. What factors appear to be associated with successful or unsuccessful AI adoption?

---

## Pipeline

### Stage 1 — Exploratory Data Analysis (`01_EDA.ipynb`)

Initial profiling of the raw dataset, including language distribution, temporal coverage, text length distribution, missing value assessment, and duplicate detection. Manual inspection of a random sample revealed significant web-scraping artifacts (navigation menus, cookie notices, subscription prompts) requiring systematic cleaning.

### Stage 2 — Data Cleaning and Filtering (`02_Clean_Filter.ipynb`)

A multi-step cleaning pipeline was applied to each article:

- HTML entity decoding, Unicode normalization, and control character removal
- Paragraph-level scoring and extraction to isolate article body text from boilerplate
- Repeated title prefix removal and consecutive duplicate line suppression
- Multi-rule quality filtering: minimum character length (≥ 300), domain blacklist, error page detection, aggregation page detection, and English alphabet ratio threshold
- Deduplication by URL and by exact cleaned text

### Stage 3 — Topic Modeling (`03_topic_modeling.ipynb`)

TF-IDF vectorization was applied using a custom stopword list of 346 terms (including domain-saturating terms such as *ai*, *technology*, and *said*), with a vocabulary of 20,000 features, a document frequency range of [20, 0.8], and unigram/bigram tokenization. K-Means clustering was evaluated over K = 10–20 using silhouette score on a 10,000-document sample. K = 20 was selected as the optimal configuration. All 20 clusters were manually inspected; 14 were retained as interpretable topics. Retained topics were mapped to industry labels and aggregated into five macro-topics: AI Applications, AI Platforms, AI Policy, AI Hardware, and AI Research.

### Stage 3b — Topic Modeling Upgrade: BERTopic (`scripts/bertopic_topics.py`, `scripts/map_industries.py`)

Stage 3 was re-done with BERTopic and compared with the original K-Means topics on the same documents. Each article was embedded with `all-MiniLM-L6-v2` (sentence-transformers), reduced with UMAP (15 neighbours, 5 dimensions, cosine), clustered with HDBSCAN (minimum cluster size 150), and described with c-TF-IDF using the same 346-term stopword list as Stage 3. Outlier articles were reassigned to their nearest topic by c-TF-IDF similarity (threshold 0.1), and the 103 resulting topics were merged down to 40 for labelling. Seeds are fixed at 42.

**A first run on the full cleaned text clustered by site template instead of subject.** Of 136 topics, several were single-site templates: PR Newswire navigation menus, MEXC crypto price pages, mql5 trading-robot listings, an AI-wallpaper gallery, Coursera course pages and a job board. K-Means had marked the wallpaper gallery as noise and put 58% of the MEXC pages in a crypto topic. However, it placed most mql5, Coursera and job-board pages in content topics such as *Generative AI Applications* and *Machine Learning and Data Science*. The second run therefore:

- removed five listing and storefront domains (5,430 articles)
- represented each article by its cleaned title plus its AI-relevant sentences from Stage 5, dropping the boilerplate around them (articles with no such sentence were dropped)
- stripped repeated site names from titles

| Corpus step | Articles |
|---|---|
| Cleaned corpus (Stage 2) | 115,006 |
| After removing listing domains | 109,576 |
| With at least one AI-relevant sentence (final corpus) | 95,467 |

**Topic quality on the same 95,467 articles.** Coherence was computed with `gensim` on the same tokenised documents, using the top 10 words of each topic. Diversity is the share of unique words across all topics' top-10 lists.

| Model | Topics | NPMI | C_v | Diversity | Largest topic (share of articles) |
|---|---|---|---|---|---|
| TF-IDF + K-Means (Stage 3, topics kept) | 14 | 0.163 | 0.721 | 0.874 | 32.1% |
| BERTopic, all topics | 103 | 0.229 | 0.706 | 0.605 | 6.5% |
| **BERTopic, merged to 40 (used below)** | 40 | **0.228** | 0.698 | 0.706 | **11.7%** |
| BERTopic, merged to 14 | 14 | 0.309 | 0.754 | 0.686 | 63.6% |

At 40 topics BERTopic raises NPMI coherence from 0.163 to 0.228. C_v is slightly lower, and diversity is lower because related topics share words (*chatgpt*, *openai*). Merging down to 14 topics gives the best coherence numbers but places 63.6% of articles in one topic, so that model was not used. After outlier reassignment, 29.8% of articles still belong to no topic (39.9% before).

**Industry mapping.** Each of the 40 topics was labelled by hand from its top words, sample titles and top domains (`results/bertopic_v2/topic_industry_map.csv`, which records the evidence for each label). This gave 34 content topics in 26 sectors, plus 6 noise topics:

- PR Newswire menus
- comment-form boilerplate
- car-insurance quote spam
- a SafetyLit citation digest
- a tabloid site template
- Indian news where "AI" is the abbreviation for Air India

Noise topics hold 3.4% of articles; 66.8% of articles receive a sector.

| Sector (top 12 of 26) | Articles | Share of assigned |
|---|---|---|
| Enterprise Software & Security | 11,187 | 17.5% |
| Semiconductors & AI Stocks | 8,204 | 12.9% |
| Government & Regulation | 6,752 | 10.6% |
| Healthcare | 4,885 | 7.7% |
| Consumer AI Apps | 4,053 | 6.4% |
| AI Labs & Leadership | 3,672 | 5.8% |
| Big Tech Platforms | 3,320 | 5.2% |
| Emerging Markets & Public Sector | 2,712 | 4.2% |
| Education | 2,361 | 3.7% |
| China & AI Geopolitics | 2,125 | 3.3% |
| Consumer Electronics | 2,113 | 3.3% |
| Elections & Misinformation | 1,917 | 3.0% |

The full list is in `results/bertopic_v2/industry_summary_bertopic.csv`.

**What changed compared with K-Means.** Stage 3 used 12 industry labels plus noise, and placed 42.5% of these articles in the general "Software / AI Platforms" label. BERTopic moved 30.6% of those 40,536 articles (12,424) to a sector outside technology. The largest such sectors were:

- Government & Regulation: 2,433 articles
- Education: 2,035 articles (no label of its own in Stage 3)
- Healthcare: 1,014 articles
- Elections & Misinformation: 966 articles
- Film, TV & Gaming: 891 articles
- Sports: 730 articles

A further 36.7% of the general bucket became BERTopic outliers. This count is conservative: an article that moved into another technology sector, such as Consumer AI Apps or Semiconductors, is not counted as moved.

Reproduce (the instructor-provided corpus is not included in this repository; place it under `data/`):

```bash
pip install -r requirements-bertopic.txt
python scripts/bertopic_topics.py --source clean --out results/bertopic          # first run, full cleaned text
python scripts/bertopic_topics.py --source hybrid --reduce-to 40,14 --out results/bertopic_v2
python scripts/map_industries.py
```

A full run takes about 7–13 minutes on an Apple M-series laptop with MPS. Embeddings are cached, so later runs are faster.

### Stage 4 — Named Entity Extraction (`04_entity_extraction.ipynb`)

Two extraction approaches were used in parallel:

- **Organization extraction**: spaCy `en_core_web_sm` NER (ORG label), followed by a normalization map (e.g., *Alphabet* → *Google*), a heuristic company-name filter, and a blacklist for generic terms
- **Technology extraction**: Taxonomy-based keyword matching across 11 technology and impact categories — Automation, Augmentation, Workflow Redesign, Cost Reduction / Efficiency, Robotics / Physical Automation, Governance / Compliance, Generative AI / LLM, AI Infrastructure / Chips, AI Services / Cloud, AI Research / Development, and Data / Model Training

### Stage 5 — Sentiment Model Training and Inference (`05_sentiment_train_apply.ipynb`)

Sentence segmentation was applied to all cleaned articles, yielding 2,510,523 sentences. A filtering step retained 596,714 sentences containing AI-related or impact-related keywords with sufficient length (≥ 40 characters, ≥ 8 words). A priority scoring system was used to select 2,000 high-information sentences for annotation.

Sentences were pre-labeled using the OpenAI API (GPT) as an annotation assistance tool, then reviewed. Labels follow a three-class scheme: *positive*, *negative*, and *uncertain/mixed*. The labeled dataset was split 70/15/15 into training, validation, and test sets.

A `roberta-base` model was fine-tuned for sequence classification using the Hugging Face `transformers` library. Training used stratified splits, standard cross-entropy loss, and epoch-level evaluation. Final performance on the held-out test set:

| Metric | Value |
|---|---|
| Accuracy | 82.3% |
| Macro F1 | 0.764 |
| Macro Precision | 0.762 |
| Macro Recall | 0.770 |

The trained model was applied to all 596,714 filtered sentences, producing probability scores for each class. Results were aggregated at the document level, topic level, and entity level, with monthly time-series outputs for trend analysis.

### Stage 6 — Visualization (`06_Visualization.ipynb`)

Sentiment trends were visualized across macro-topics, individual topics, top-mentioned companies, and AI technology categories over time.

---

## Repository Structure

```
├── NLP_Final_Project_01_EDA.ipynb
├── NLP_Final_Project_02_Clean_Filter.ipynb
├── NLP_Final_Project_03_topic_modeling.ipynb
├── NLP_Final_Project_04_entity_extraction.ipynb
├── NLP_Final_Project_05_sentiment_train_apply.ipynb
├── NLP_Final_Project_06_Visualization.ipynb
├── scripts/
│   ├── bertopic_topics.py          # Stage 3b: BERTopic + coherence comparison with K-Means
│   └── map_industries.py           # Stage 3b: topic -> sector mapping and summary
├── results/
│   ├── bertopic/                   # first run on full cleaned text (site-template topics)
│   └── bertopic_v2/                # final run: metrics, 40-topic table, sector map and summary
├── requirements-bertopic.txt
└── README.md
```

---

## Dependencies

```
pandas, numpy, pyarrow
scikit-learn
spacy (en_core_web_sm)
transformers, datasets, accelerate, evaluate
matplotlib, seaborn, wordcloud
openai
tqdm
```

Install via:
```bash
pip install pandas numpy pyarrow scikit-learn spacy transformers datasets accelerate evaluate matplotlib seaborn wordcloud openai tqdm
python -m spacy download en_core_web_sm
```

---

## Reproducibility Notes

- Notebooks must be executed in order (01 → 06); each stage depends on Parquet/CSV outputs from the previous stage.
- Random seeds are fixed at 42 throughout.
- Notebook 05 was originally developed in Google Colab with Google Drive as the data directory. The `DATA_DIR` path variable should be updated for local execution.
- The OpenAI API was used solely for annotation pre-labeling assistance (Stage 5). It is not required to reproduce model training or inference; the labeled dataset is saved as `sent_annotation_labeled_2000.csv`.

---

## Limitations

- The sentiment model was trained on 2,000 sentences, which limits its generalizability across less-represented topics and domains.
- Organization normalization is rule-based and may introduce errors for ambiguous entity names.
- Topic clusters were assigned labels through manual inspection, introducing subjectivity into the topic taxonomy. This also applies to the BERTopic sector map in Stage 3b, which records the evidence for each label so it can be checked.
- In Stage 3b, 29.8% of articles remain BERTopic outliers and receive no sector. The sector shares describe the other 66.8% of articles.
- Document-level topic assignments are not published, because they contain URLs and titles from the instructor-provided corpus. Only aggregate tables are included.
- The corpus is limited to English-language articles and may not reflect non-Anglophone AI discourse.
