import argparse
import numpy as np
import os
import display

from train import train
from evaluation import run_evaluation
from visualizer import reduce_to_3d, plot_3d


def save_embeddings(W_in, W_out, word2idx, idx2word):
    os.makedirs('embeddings', exist_ok=True)
    np.save('embeddings/W_in.npy', W_in)
    np.save('embeddings/W_out.npy', W_out)
    np.save('embeddings/word2idx.npy', word2idx)
    np.save('embeddings/idx2word.npy', idx2word)
    display.saved()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-size", type=int, default=10_000_000,
        help="Max characters to read from text8 (default: 10_000_000)"
    )
    args = parser.parse_args()
    if args.corpus_size <= 0:
        parser.error("--corpus-size must be a positive integer")

    W_in, W_out, word2idx, idx2word = train(epochs=3,
                                            corpus_size=args.corpus_size)
    save_embeddings(W_in, W_out, word2idx, idx2word)

    # load saved embeddings (comment out the 3 lines above
    # and save_embeddings()) and uncomment 5 lines below
    #
    # W_in = np.load('embeddings/W_in.npy')
    # W_out = np.load('embeddings/W_out.npy')
    # word2idx = np.load('embeddings/word2idx.npy', allow_pickle=True).item()
    # idx2word = np.load('embeddings/idx2word.npy', allow_pickle=True).item()

    run_evaluation(W_in, word2idx, idx2word)

    answer = input("\nRun t-SNE visualization? [Y/n]: ").strip().lower()
    if answer in ("", "y", "yes"):
        reduced = reduce_to_3d(W_in, n_words=2000)
        plot_3d(reduced, idx2word)
