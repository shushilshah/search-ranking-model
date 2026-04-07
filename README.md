# Search Ranking Model

A Learning-to-Rank (LTR) ML project for web/document search using Python + XGBoost.

## Project structure

```
search_ranking/
├── configs/            # YAML configs for features, model, training
├── data/
│   ├── raw/            # Raw query-document pairs + click logs
│   ├── processed/      # Cleaned, joined datasets
│   └── features/       # Computed feature matrices
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── features/       # Feature extraction (BM25, TF-IDF, signals)
│   ├── models/         # LTR model: pointwise, pairwise (LambdaMART)
│   ├── evaluation/     # NDCG, MRR, MAP metrics
│   └── serving/        # FastAPI inference endpoint
├── notebooks/          # EDA and experiment notebooks
├── tests/              # Unit tests
└── outputs/            # Saved models, reports
```


## LTR approaches included

| Approach | Algorithm | Use case |
|---|---|---|
| Pointwise | XGBoost regressor | Binary/graded relevance labels |
| Pairwise | XGBoost LambdaMART (`rank:pairwise`) | Pairwise preference data |
| Listwise | XGBoost LambdaMART (`rank:ndcg`) | Maximise NDCG directly |

## Key metrics

- **NDCG@k** — primary ranking quality metric
- **MRR** — mean reciprocal rank (good for navigational queries)
- **MAP** — mean average precision (precision-focused)