import numpy as np


def build_noise_distribution(
    counts: dict,
    word2idx: dict
) -> np.ndarray:
    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size)
    for word, idx in word2idx.items():
        freqs[idx] = counts[word]
    powered = freqs ** 0.75
    noise_dist = powered / powered.sum()
    return noise_dist


def sample_negatives(
    noise_dist: np.ndarray,
    k: int,
    exclude_idx: int
) -> np.ndarray:
    negatives = []
    while len(negatives) < k:
        candidates = np.random.choice(
            len(noise_dist), size=k * 2, p=noise_dist
        )
        for c in candidates:
            if c != exclude_idx and len(negatives) < k:
                negatives.append(c)
    return np.array(negatives, dtype=np.int32)
