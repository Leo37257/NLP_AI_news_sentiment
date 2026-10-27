"""Map the 40 BERTopic (v2) topics to industries, flag noise topics, and compare with the original K-Means labels.

The mapping is manual, like the original stage 3: each topic was labelled from its top words, sample titles and
top domains (see the `evidence` column). Inputs come from scripts/bertopic_topics.py --source hybrid --reduce-to 40.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bertopic_v2"
NOISE = "Noise"
MAP = {  # topic: (industry, evidence)
    0: ("Enterprise Software & Security", "data, generative, security, customer, automation, agents, cloud"),
    1: ("Semiconductors & AI Stocks", "nvidia, stock, etf, shares, computing"),
    2: ("Government & Regulation", "risks, regulation, eu, safety, white house, biden"),
    3: ("Healthcare", "health, medical, patients, clinical, cancer, drug"),
    4: ("Consumer AI Apps", "chatgpt, users, gpt4, chatbot, app, privacy"),
    5: ("AI Labs & Leadership", "openai, altman, musk, ceo, board"),
    6: ("Emerging Markets & Public Sector", "india, uae, dubai, africa, minister; Saudi/Indian digital projects"),
    7: ("Education", "students, teachers, schools, educators"),
    8: ("Big Tech Platforms", "google, gemini, bard, claude, coding"),
    9: ("China & AI Geopolitics", "deepseek, china, baidu, alibaba, chips"),
    10: ("Consumer Electronics", "apple, samsung, galaxy, iphone, siri"),
    11: ("Elections & Misinformation", "election, trump, misinformation, deepfakes, voters"),
    12: ("Automotive & Computer Vision", "vision, automotive, vehicles, driving, traffic"),
    13: ("Energy & Climate", "energy, power, climate, data centers, grid"),
    14: ("Data Science & Analytics", "machine learning, data science, ml, analytics, MLOps"),
    15: (NOISE, "PR Newswire navigation menu (overviewview, consumer products, heavy industry): 86% one site template"),
    16: ("Retail & Marketing", "marketing, shopping, fashion, retail, brands"),
    17: ("Film, TV & Gaming", "actors, hollywood, strike, studios, video game"),
    18: ("Big Tech Platforms", "copilot, microsoft, windows, bing, 365"),
    19: ("Creative & Video Tools", "adobe, video, creative; 44% heygen.com landing pages"),
    20: ("Sports", "nfl, sports, players, football, predictions"),
    21: ("Music", "music industry, songwriters, suno, musicians"),
    22: (NOISE, "comment-form boilerplate (save email website browser comment)"),
    23: (NOISE, "'AI' = Air India abbreviation plus mixed Indian news (flights, cricket predictions)"),
    24: (NOISE, "car-insurance quote spam (compareautoinsurance)"),
    25: ("AI Labs & Leadership", "grok, xai, elon musk"),
    26: (NOISE, "Daily Mail / Daily Star tabloid template, mixed subjects"),
    27: ("Consumer AI Apps", "image generation, dalle, ghibli"),
    28: ("Scientific Research", "materials, quantum, physics, scientists"),
    29: ("Consumer AI Apps", "AI companions, dating, relationships"),
    30: ("Religion & Ethics", "pope, vatican, church, ethics"),
    31: ("Philanthropy & Nonprofits", "philanthropy, foundations, nonprofits, funding"),
    32: ("Music", "beatles, mccartney, lennon's voice"),
    33: ("Defense & Military", "gaza, israeli military, targets"),
    34: ("Legal Services", "lawyers, court, bogus cases"),
    35: (NOISE, "SafetyLit weekly citation digest template"),
    36: ("Public Safety", "wildfires, smoke, firefighters, detection cameras"),
    37: ("AI Labs & Leadership", "scarlett johansson, chatgpt voice"),
    38: ("Consumer AI Apps", "browser, chatgpt atlas, opera"),
    39: ("Public Safety", "gun detection, zeroeyes, schools"),
}
GENERIC_ORIGINAL = "Software / AI Platforms"
# Industries that are still "tech" in the broad sense. A generic-bucket doc that lands in one of these is not
# counted as moved to a specific sector, so the "moved" share below is a conservative number.
TECH_LIKE = {"Enterprise Software & Security", "Big Tech Platforms", "Consumer AI Apps", "AI Labs & Leadership",
             "Data Science & Analytics", "Creative & Video Tools", "Semiconductors & AI Stocks"}


def main() -> None:
    info = pd.read_csv(OUT / "topic_info_40.csv")
    docs = pd.read_parquet(OUT / "doc_topics_with_original_labels.parquet")
    assert set(info.Topic) - {-1} == set(MAP), "mapping must cover every topic"

    tmap = pd.DataFrame([{"topic": k, "industry": v[0], "evidence": v[1]} for k, v in MAP.items()])
    tmap = tmap.merge(info.rename(columns={"Topic": "topic", "Count": "docs", "top_words": "top_words"})[
        ["topic", "docs", "top_words"]], on="topic")
    tmap.sort_values("topic").to_csv(OUT / "topic_industry_map.csv", index=False)

    docs["industry_bertopic"] = docs["bertopic_topic_40"].map(lambda t: MAP[t][0] if t in MAP else "Unassigned (outlier)")
    n = len(docs)
    content = docs[~docs.industry_bertopic.isin([NOISE, "Unassigned (outlier)"])]
    ind = content.industry_bertopic.value_counts().rename_axis("industry").reset_index(name="docs")
    ind["share_of_assigned"] = (ind.docs / len(content)).round(4)
    ind.to_csv(OUT / "industry_summary_bertopic.csv", index=False)

    generic = docs[docs.industry == GENERIC_ORIGINAL]
    gen_specific = generic[~generic.industry_bertopic.isin(TECH_LIKE | {NOISE, "Unassigned (outlier)"})]
    summary = {
        "documents": n,
        "content_topics": int(sum(1 for v in MAP.values() if v[0] != NOISE)),
        "noise_topics": int(sum(1 for v in MAP.values() if v[0] == NOISE)),
        "industries": int(ind.shape[0]),
        "share_assigned_to_industry": round(len(content) / n, 4),
        "share_noise_topics": round(float((docs.industry_bertopic == NOISE).mean()), 4),
        "share_outliers": round(float((docs.industry_bertopic == "Unassigned (outlier)").mean()), 4),
        "original_generic_bucket": {"label": GENERIC_ORIGINAL, "docs": int(len(generic)),
                                    "moved_to_a_non_tech_sector": int(len(gen_specific)),
                                    "share_moved": round(len(gen_specific) / max(len(generic), 1), 4),
                                    "share_outlier": round(float((generic.industry_bertopic == "Unassigned (outlier)").mean()), 4)},
        "original_industries_in_these_docs": int(docs.industry.nunique()),
    }
    (OUT / "industry_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(ind.to_string(index=False))


if __name__ == "__main__":
    main()
