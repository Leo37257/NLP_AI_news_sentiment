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

### Stage 4 — Named Entity Extraction (`04_entity_extraction.ipynb`)

Two extraction approaches were used in parallel:

- **Organization extraction**: spaCy `en_core_web_sm` NER (ORG label), followed by a normalization map (e.g., *Alphabet* → *Google*), a heuristic company-name filter, and a blacklist for generic terms
- **Technology extraction**: Taxonomy-based keyword matching across six categories — Automation, Augmentation, Generative AI, Autonomous Systems, Computer Vision, and Natural Language Processing

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
- Topic clusters were assigned labels through manual inspection, introducing subjectivity into the topic taxonomy.
- The corpus is limited to English-language articles and may not reflect non-Anglophone AI discourse.
