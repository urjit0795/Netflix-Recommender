"""
Semantic search for movies using Sentence Transformer embeddings.

Given a free-text query (e.g. "gritty crime thriller from the 90s"), this
module encodes the query and every movie's metadata (title + genre + year)
into the same embedding space using a pretrained Sentence Transformer, then
ranks movies by cosine similarity to the query embedding.

This complements the existing TF-IDF based content recommender: TF-IDF only
matches on exact/overlapping vocabulary, while sentence embeddings capture
semantic meaning (e.g. "space opera" can match "Sci-Fi" movies even without
shared keywords).

Embeddings are cached to disk (as a .npy file) alongside a hash of the
underlying movie content, so they are only recomputed when the dataset
actually changes -- not on every app restart.
"""

import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = os.path.join("data", "embeddings")
EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "movie_embeddings.npy")
HASH_PATH = os.path.join(EMBEDDINGS_DIR, "movie_embeddings.hash")


def _build_movie_text(movies_df: pd.DataFrame) -> pd.Series:
    """Combine title, genre, and (optionally) description into one text field."""
    text = (
        movies_df["title"].fillna("") + ". "
        + movies_df["genre"].fillna("") + " movie released in "
        + movies_df["release_year"].astype(str) + "."
    )

    if "description" in movies_df.columns:
        text = text + " " + movies_df["description"].fillna("")

    return text


def _content_hash(movie_text: pd.Series) -> str:
    """Hash the movie content so we know when cached embeddings are stale."""
    joined = "|".join(movie_text.tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def load_embedding_model(model_name: str = MODEL_NAME):
    """Lazily import and load the Sentence Transformer model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def get_movie_embeddings(movies_df: pd.DataFrame, model=None) -> np.ndarray:
    """
    Return embeddings for every movie in ``movies_df``.

    Uses a disk cache (data/embeddings/movie_embeddings.npy) keyed by a hash
    of the movie content, so embeddings are only recomputed when the
    underlying movie metadata changes.
    """
    movie_text = _build_movie_text(movies_df)
    current_hash = _content_hash(movie_text)

    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(HASH_PATH):
        with open(HASH_PATH, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()

        if cached_hash == current_hash:
            return np.load(EMBEDDINGS_PATH)

    if model is None:
        model = load_embedding_model()

    embeddings = model.encode(
        movie_text.tolist(),
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(HASH_PATH, "w", encoding="utf-8") as f:
        f.write(current_hash)

    return embeddings


def semantic_search(
    query: str,
    movies_df: pd.DataFrame,
    movie_embeddings: np.ndarray,
    model=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank movies by semantic similarity to a free-text query.

    Returns the top_n most similar movies with an added ``similarity`` column.
    """
    if not query or not query.strip():
        return pd.DataFrame()

    if model is None:
        model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    scores = cosine_similarity(query_embedding, movie_embeddings)[0]

    result = movies_df.copy()
    result["similarity"] = scores

    return result.sort_values("similarity", ascending=False).head(top_n)
