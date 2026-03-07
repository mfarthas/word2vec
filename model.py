import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def init_embeddings(
    vocab_size: int,
    embed_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    W_in = (np.random.random((vocab_size, embed_dim)) - 0.5) / embed_dim
    W_out = np.zeros((vocab_size, embed_dim))
    return W_in, W_out
