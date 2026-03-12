import time
import display

from corpus import load_corpus
from vocab import build_vocab, tokens_to_indices
from subsampling import subsample
from noise_distribution import build_noise_distribution, sample_negatives
from training_pairs import generate_pairs
from model import init_embeddings
from forward_backward_pass import step


def get_lr(
    lr_start: float,
    lr_min: float,
    current_step: int,
    total_steps: int
) -> float:
    lr = lr_start - (lr_start - lr_min) * (current_step / total_steps)
    return max(lr, lr_min)


def train(
    epochs: int = 3,
    embed_dim: int = 100,
    window_size: int = 5,
    k: int = 5,
    lr_start: float = 0.025,
    lr_min: float = 0.0001,
    corpus_size: int = 10_000_000
):
    display.training_header(
        epochs, embed_dim, window_size, k, lr_start, lr_min
    )

    tokens = load_corpus(max_chars=corpus_size)
    word2idx, idx2word, counts = build_vocab(tokens)
    indices = tokens_to_indices(tokens, word2idx)
    display.corpus_loaded(len(tokens), len(word2idx))

    total_tokens = len(indices)
    subsampled = subsample(
        indices, counts, word2idx, idx2word, total_tokens
    )
    display.subsampling_done(100 * len(subsampled) / len(indices))

    noise_dist = build_noise_distribution(counts, word2idx)
    display.noise_done()

    W_in, W_out = init_embeddings(len(word2idx), embed_dim)
    display.embeddings_init(W_in.shape, W_in.nbytes / 1024 / 1024)

    total_steps = epochs * len(subsampled) * window_size
    global_step = 0

    for epoch in range(epochs):
        display.epoch_header(epoch + 1, epochs)
        pairs = generate_pairs(subsampled, window_size)
        epoch_loss = 0.0
        t0 = time.time()
        n_pairs = len(pairs)

        with display.make_progress(lr_start) as progress:
            task = progress.add_task(
                f"epoch {epoch + 1}",
                total=n_pairs,
                loss=0.0,
                lr=lr_start,
                speed=0.0
            )
            for i, (center_idx, context_idx) in enumerate(pairs):
                neg_indices = sample_negatives(
                    noise_dist, k, context_idx
                )
                lr = get_lr(lr_start, lr_min, global_step, total_steps)
                loss = step(
                    center_idx, context_idx,
                    neg_indices, W_in, W_out, lr
                )
                epoch_loss += loss
                global_step += 1

                if (i + 1) % 5_000 == 0:
                    elapsed = time.time() - t0
                    progress.update(
                        task,
                        advance=5_000,
                        loss=epoch_loss / (i + 1),
                        lr=lr,
                        speed=(i + 1) / elapsed
                    )

        display.epoch_done(epoch_loss / n_pairs, time.time() - t0)

    return W_in, W_out, word2idx, idx2word
