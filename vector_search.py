"""
FAISS-backed vector search for movies.

This upgrades the brute-force cosine-similarity search in
``semantic_search.py`` (plain sklearn, O(n) scan over every movie every
query) to an actual vector index using FAISS -- the same class of tool
production recommender/retrieval systems (and RAG pipelines) use to search
over millions/billions of embeddings in milliseconds.

At this dataset's scale (100 movies) a brute-force scan is already fast, so
this isn't about raw speed here -- it's about adopting the right retrieval
primitive before the dataset grows and before wiring this into a RAG
pipeline, where an in-process vector index is the standard building block.

Since movie/query embeddings are L2-normalized (see
semantic_search.get_movie_embeddings), cosine similarity is equivalent to
inner product, so this uses a flat inner-product index (``IndexFlatIP``) --
an *exact* nearest-neighbor search, not an approximate one. Swapping to an
approximate index (e.g. IndexIVFFlat, HNSW) is a later scaling step once the
catalog is large enough to need it -- not needed yet at 100 movies, and
premature to add now.
"""

import os

import faiss
import numpy as np
import pandas as pd

from semantic_search import (
    EMBEDDINGS_DIR,
    HASH_PATH,
    _build_movie_text,
    _content_hash,
    get_movie_embeddings,
    load_embedding_model,
)

INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "movie_faiss.index")
INDEX_HASH_PATH = os.path.join(EMBEDDINGS_DIR, "movie_faiss.hash")


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    """Build a flat inner-product FAISS index over the given embeddings."""
    embeddings = np.ascontiguousarray(embeddings.astype("float32"))
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def get_faiss_index(movies_df: pd.DataFrame, model=None) -> "faiss.Index":
    """
    Return a FAISS index over every movie's embedding, cached to disk
    (data/embeddings/movie_faiss.index) keyed by the same content hash used
    for the embeddings cache, so it's only rebuilt when the dataset changes.
    """
    movie_text = _build_movie_text(movies_df)
    current_hash = _content_hash(movie_text)

    if os.path.exists(INDEX_PATH) and os.path.exists(INDEX_HASH_PATH):
        with open(INDEX_HASH_PATH, "r", encoding="utf-8") as f:
            cached_hash = f.read().strip()

        if cached_hash == current_hash:
            return faiss.read_index(INDEX_PATH)

    embeddings = get_movie_embeddings(movies_df, model=model)
    index = build_faiss_index(embeddings)

    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(INDEX_HASH_PATH, "w", encoding="utf-8") as f:
        f.write(current_hash)

    return index


def faiss_search(
    query: str,
    movies_df: pd.DataFrame,
    index: "faiss.Index",
    model=None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank movies by semantic similarity to a free-text query, using the FAISS
    index for retrieval instead of a brute-force sklearn scan.

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
    query_embedding = np.ascontiguousarray(query_embedding.astype("float32"))

    top_n = min(top_n, index.ntotal)
    scores, indices = index.search(query_embedding, top_n)

    result = movies_df.iloc[indices[0]].copy()
    result["similarity"] = scores[0]

    return result
