import numpy as np
import display


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
    return np.dot(v1, v2) / denom


def most_similar(
    word: str,
    W_in: np.ndarray,
    word2idx: dict,
    idx2word: dict,
    top_n: int = 5
) -> list[tuple[str, float]]:
    if word not in word2idx:
        return []
    vec = W_in[word2idx[word]]
    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-10
    W_norm = W_in / norms
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    similarities = W_norm @ vec_norm
    top_idx = np.argsort(similarities)[::-1]
    results = []
    for idx in top_idx:
        if idx2word[idx] != word:
            results.append((idx2word[idx], float(similarities[idx])))
        if len(results) >= top_n:
            break
    return results


def analogy(
    a: str,
    b: str,
    c: str,
    W_in: np.ndarray,
    word2idx: dict,
    idx2word: dict,
    top_n: int = 3
) -> list[tuple[str, float]]:
    for w in [a, b, c]:
        if w not in word2idx:
            return []
    vec = W_in[word2idx[b]] - W_in[word2idx[a]] + W_in[word2idx[c]]
    norms = np.linalg.norm(W_in, axis=1, keepdims=True) + 1e-10
    W_norm = W_in / norms
    vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
    similarities = W_norm @ vec_norm
    exclude = {word2idx[a], word2idx[b], word2idx[c]}
    top_idx = np.argsort(similarities)[::-1]
    results = []
    for idx in top_idx:
        if idx not in exclude:
            results.append((idx2word[idx], float(similarities[idx])))
        if len(results) >= top_n:
            break
    return results


def run_evaluation(
    W_in: np.ndarray,
    word2idx: dict,
    idx2word: dict
):
    test_words = ['king', 'man', 'woman', 'city', 'one', 'war', 'time']
    neighbors = [
        (word, most_similar(word, W_in, word2idx, idx2word))
        for word in test_words
        if word in word2idx
    ]
    display.neighbors_table(neighbors)

    tests = [
        ("man", "king", "woman"),
        ("paris", "france", "berlin"),
        ("one", "two", "three"),
    ]
    analogies = [
        (a, b, c, analogy(a, b, c, W_in, word2idx, idx2word))
        for a, b, c in tests
    ]
    display.analogies_table(analogies)
