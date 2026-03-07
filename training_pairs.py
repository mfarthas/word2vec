import numpy as np


def generate_pairs(
    indices: list[int],
    window_size: int = 5
) -> list[tuple[int, int]]:
    pairs = []
    corpus_len = len(indices)
    for i, center in enumerate(indices):
        actual_window = np.random.randint(1, window_size + 1)
        start = max(0, i - actual_window)
        end = min(corpus_len, i + actual_window + 1)
        for j in range(start, end):
            if j == i:
                continue
            pairs.append((center, indices[j]))
    return pairs
