from collections import Counter


def build_vocab(tokens: list[str], min_count: int = 5):
    counts = Counter(tokens)
    counts = {w: c for w, c in counts.items() if c >= min_count}
    sorted_words = sorted(counts.keys(), key=lambda w: -counts[w])
    word2idx = {w: i for i, w in enumerate(sorted_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, counts


def tokens_to_indices(
    tokens: list[str],
    word2idx: dict
) -> list[int]:
    return [word2idx[t] for t in tokens if t in word2idx]
