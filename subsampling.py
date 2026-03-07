import numpy as np


def keep_probability(
    word_freq: int,
    total_tokens: int,
    threshold: float = 1e-3
) -> float:
    z = word_freq / total_tokens
    prob = (np.sqrt(z / threshold) + 1) * (threshold / z)
    return min(prob, 1.0)


def subsample(
    indices: list[int],
    counts: dict,
    word2idx: dict,
    idx2word: dict,
    total_tokens: int,
    threshold: float = 1e-3
) -> list[int]:
    kept = []
    for idx in indices:
        word = idx2word[idx]
        freq = counts[word]
        prob = keep_probability(freq, total_tokens, threshold)
        if np.random.random() < prob:
            kept.append(idx)
    return kept
