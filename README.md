# Word2Vec — Skip-gram with Negative Sampling

## Setup

```bash
pip install -r requirements.tx
python word2vec.py
```

Downloads ~30MB of training data (text8) on first run.

## How it works

| File | Role |
|---|---|
| `corpus.py` | Downloads and tokenizes text8 |
| `vocab.py` | Builds vocabulary, filters rare words |
| `subsampling.py` | Drops frequent words probabilistically |
| `noise_distribution.py` | Unigram distribution for negative sampling |
| `training_pairs.py` | Generates skip-gram (center, context) pairs |
| `model.py` | Initializes embedding matrices W_in and W_out |
| `forward_backward_pass.py` | Forward pass, loss, gradients, SGD update |
| `train.py` | Training loop with linear lr decay |
| `evaluation.py` | Nearest neighbors and word analogies |
| `visualizer.py` | t-SNE 3D projection via Plotly |
| `display.py` | All Rich terminal output |
| `word2vec.py` | Entry point — runs everything |

Trained embeddings are saved to `embeddings/` and can be loaded
directly to skip retraining.