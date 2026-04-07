import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

QUERIES = [
    "machine learning tutorial",
    "python web scraping",
    "neural network backpropagation",
    "docker containerisation guide",
    "sql window functions explained",
    "git branching strategies",
    "rest api design best practices",
    "linux file permissions",
    "kubernetes pod scheduling",
    "data engineering pipeline",
]

DOCS_PER_QUERY = 20  # candidate documents per query
N_QUERIES = len(QUERIES)


def _simulate_relevance(query_idx: int, doc_idx: int) -> int:
    base_prob = max(0.0, 1.0 - doc_idx / DOCS_PER_QUERY)
    score = RNG.uniform(0, 1) * base_prob + RNG.uniform(0, 0.3)
    if score > 0.75:
        return 3
    elif score > 0.5:
        return 2
    elif score > 0.25:
        return 1
    return 0


def generate_dataset(output_path: str = "data/raw/queries_docs.csv") -> pd.DataFrame:
    rows = []

    for q_idx, query in enumerate(QUERIES):
        qid = f"q{q_idx:04d}"
        query_length = len(query.split())
        query_idf_sum = RNG.uniform(5, 25)
        is_navigational = int(RNG.uniform() > 0.8)

        for d_idx in range(DOCS_PER_QUERY):
            doc_id = f"d{q_idx:04d}_{d_idx:04d}"
            relevance = _simulate_relevance(q_idx, d_idx)

            rel_signal = relevance / 3.0
            def noise(scale=0.3): return RNG.normal(0, scale)

            row = {
                "qid": qid,
                "doc_id": doc_id,
                "query": query,
                "relevance": relevance,
                # Text similarity features
                "bm25_score": max(0, rel_signal * 20 + noise(4)),
                "tfidf_cosine": np.clip(rel_signal + noise(0.2), 0, 1),
                "query_term_coverage": np.clip(rel_signal + noise(0.15), 0, 1),
                "title_match_score": np.clip(rel_signal + noise(0.25), 0, 1),
                "body_match_score": np.clip(rel_signal + noise(0.2), 0, 1),
                # Document features
                "doc_length": max(100, int(500 + RNG.normal(0, 300))),
                "doc_pagerank": max(0, RNG.exponential(0.5) + rel_signal * 0.3),
                "doc_freshness_days": max(0, int(RNG.exponential(200))),
                "avg_click_rate": np.clip(rel_signal * 0.4 + noise(0.1), 0, 1),
                # Query features
                "query_length": query_length,
                "query_idf_sum": query_idf_sum,
                "is_navigational": is_navigational,
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows → {output_path}")
    print(
        f"Relevance distribution:\n{df['relevance'].value_counts().sort_index()}")
    return df


if __name__ == "__main__":
    generate_dataset()
