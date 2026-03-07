import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE


def reduce_to_3d(W_in: np.ndarray, n_words: int = 2000) -> np.ndarray:
    vectors = W_in[:n_words]
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    return tsne.fit_transform(vectors)


def plot_3d(reduced: np.ndarray, idx2word: dict):
    categories = {
        'royalty': [
            'king', 'queen', 'emperor', 'prince',
            'throne', 'crown', 'royal', 'dynasty',
            'monarch', 'reign'
        ],
        'numbers': [
            'one', 'two', 'three', 'four', 'five',
            'six', 'seven', 'eight', 'nine', 'zero'
        ],
        'time': [
            'time', 'year', 'century', 'period', 'age',
            'era', 'decade', 'history', 'ancient', 'modern'
        ],
        'war': [
            'battle', 'army', 'soldier', 'war', 'troops',
            'victory', 'defeat', 'weapons', 'attack', 'force'
        ],
    }

    word2idx = {v: k for k, v in idx2word.items()}
    rows = []
    for category, members in categories.items():
        for word in members:
            if word in word2idx and word2idx[word] < len(reduced):
                idx = word2idx[word]
                rows.append({
                    'x': reduced[idx, 0],
                    'y': reduced[idx, 1],
                    'z': reduced[idx, 2],
                    'word': word,
                    'category': category,
                })

    df = pd.DataFrame(rows)
    n_words = len(df)
    n_cats = df['category'].nunique()
    print(f"Plotting {n_words} words across {n_cats} categories")

    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='category',
        text='word',
        title='Word2Vec Embeddings — t-SNE 3D Projection',
        opacity=0.85,
    )
    fig.update_traces(marker=dict(size=6))
    fig.show()
