import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from semantic_search import load_embedding_model
from vector_search import get_faiss_index, faiss_search
from llm_reranker import (
    build_user_context,
    format_candidates,
    build_reranking_prompt
)
import rag_chat


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Recommendation System")
st.write(
    "An interactive app to explore movie recommendations using popularity-based, "
    "content-based, collaborative filtering, hybrid, and AI-powered semantic "
    "search recommendation methods."
)


@st.cache_data
def load_data():
    movies_df = pd.read_csv("data/movies.csv")
    users_df = pd.read_csv("data/users.csv")
    ratings_df = pd.read_csv("data/ratings.csv")
    return movies_df, users_df, ratings_df


movies, users, ratings = load_data()


col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Movies", f"{len(movies):,}")

with col2:
    st.metric("Users", f"{ratings['user_id'].nunique():,}")

with col3:
    st.metric("Ratings", f"{len(ratings):,}")


with st.expander("📊 Dataset Overview"):
    st.write(f"**Movies:** {len(movies):,}")
    st.write(f"**Users:** {ratings['user_id'].nunique():,}")
    st.write(f"**Ratings:** {len(ratings):,}")
    if "genre" in movies.columns:
        st.write(f"**Genres:** {movies['genre'].nunique():,}")


@st.cache_resource
def build_content_similarity(movies_df):
    movies_df = movies_df.copy()

    movies_df["content"] = (
        movies_df["title"].fillna("") + " " +
        movies_df["genre"].fillna("") + " " +
        movies_df["release_year"].astype(str)
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["content"])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


content_similarity = build_content_similarity(movies)


@st.cache_resource
def get_semantic_search_resources(movies_df):
    """Load the Sentence Transformer model and (cached) FAISS index over
    every movie's embedding. The index is rebuilt only when the underlying
    movie data changes (see vector_search.get_faiss_index)."""
    model = load_embedding_model()
    index = get_faiss_index(movies_df, model=model)
    return model, index


semantic_model, movie_index = get_semantic_search_resources(movies)


@st.cache_resource
def build_collaborative_similarity(ratings_df):
    user_movie_matrix_df = ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    similarity_matrix = cosine_similarity(user_movie_matrix_df.T)

    movie_similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_movie_matrix_df.columns,
        columns=user_movie_matrix_df.columns
    )

    return user_movie_matrix_df, movie_similarity_df


user_movie_matrix, movie_similarity_df = build_collaborative_similarity(ratings)


def recommend_popular_movies(top_n=10, min_ratings=5):
    movie_stats = (
        ratings.groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            avg_user_rating=("rating", "mean")
        )
        .reset_index()
    )

    movie_stats = movie_stats.merge(movies, on="movie_id", how="left")

    return (
        movie_stats[movie_stats["rating_count"] >= min_ratings]
        .sort_values(["avg_user_rating", "rating_count"], ascending=False)
        .head(top_n)
    )


def recommend_content_based(movie_title, top_n=10):
    matches = movies[movies["title"].str.lower().str.contains(movie_title.lower())]

    if matches.empty:
        return pd.DataFrame()

    movie_idx = matches.index[0]

    scores = list(enumerate(content_similarity[movie_idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i for i, score in scores[1:top_n + 1]]

    return movies.loc[movie_indices]


def recommend_for_user(user_id, top_n=10):
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    user_ratings = user_movie_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index.tolist()

    scores = {}

    for movie_id in watched_movies:
        rating = user_ratings[movie_id]
        similar_movies = movie_similarity_df[movie_id]

        for similar_movie_id, similarity_score in similar_movies.items():
            if similar_movie_id in watched_movies:
                continue

            scores[similar_movie_id] = (
                scores.get(similar_movie_id, 0) + similarity_score * rating
            )

    if not scores:
        return pd.DataFrame()

    recommended_ids = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return movies[movies["movie_id"].isin(recommended_ids)]


def hybrid_recommend(user_id, movie_title, top_n=10, content_weight=0.5):
    cf_weight = 1.0 - content_weight

    # Find the selected movie
    matches = movies[
        movies["title"].str.lower().str.contains(movie_title.lower())
    ]

    # Return empty DataFrame if movie or user does not exist
    if matches.empty or user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    movie_idx = matches.index[0]
    movie_id = movies.loc[movie_idx, "movie_id"]

    # Make sure selected movie exists in collaborative similarity matrix
    if movie_id not in movie_similarity_df.columns:
        return pd.DataFrame()

    # -----------------------------
    # Content-based scores
    # -----------------------------
    content_scores = pd.Series(
        content_similarity[movie_idx],
        index=movies["movie_id"]
    )

    # -----------------------------
    # Collaborative filtering scores
    # -----------------------------
    cf_scores = movie_similarity_df[movie_id]

    # Get movies already watched by the user
    user_ratings = user_movie_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index.tolist()

    # -----------------------------
    # Combine scores
    # -----------------------------
    combined_scores = (
        content_weight * content_scores +
        cf_weight * cf_scores
    )

    # Do not recommend movies already watched by the user
    combined_scores = combined_scores.drop(
        labels=watched_movies,
        errors="ignore"
    )

    # -----------------------------
    # Rank candidate movies
    # -----------------------------
    ranked_scores = (
        combined_scores
        .sort_values(ascending=False)
        .head(top_n)
    )

    # Get movie metadata
    recommendations = movies[
        movies["movie_id"].isin(ranked_scores.index)
    ].copy()

    # Add the hybrid recommendation score
    recommendations["hybrid_score"] = (
        recommendations["movie_id"].map(ranked_scores)
    )

    # Preserve actual ranking order
    recommendations = (
        recommendations
        .sort_values(
            "hybrid_score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return recommendations

def display_recommendations(result, include_ratings=False, include_similarity=False):
    if result.empty:
        st.warning("No recommendations found. Try a different input.")
        return

    if include_ratings:
        display_cols = [
            "title",
            "genre",
            "release_year",
            "avg_user_rating",
            "rating_count"
        ]
    elif include_similarity:
        display_cols = [
            "title",
            "genre",
            "release_year",
            "similarity"
        ]
    else:
        display_cols = [
            "title",
            "genre",
            "release_year"
        ]

        # Hybrid recommendations include the score produced by
        # the combined content + collaborative filtering model.
        if "hybrid_score" in result.columns:
            display_cols.append("hybrid_score")

    available_cols = [col for col in display_cols if col in result.columns]

    display_df = result[available_cols].copy()

    # Make ranking scores easier to read in the Streamlit table.
    if "hybrid_score" in display_df.columns:
        display_df["hybrid_score"] = display_df["hybrid_score"].round(4)

    if "similarity" in display_df.columns:
        display_df["similarity"] = display_df["similarity"].round(4)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


st.sidebar.title("Recommendation Settings")
st.sidebar.write(
    "Compare different recommendation strategies using movie ratings and metadata."
)

option = st.sidebar.selectbox(
    "Choose recommendation type",
    [
        "Popular Movies",
        "Content-Based Recommendation",
        "User-Based Personalized Recommendation",
        "Hybrid Recommendation",
        "Semantic Search (AI-Powered)",
        "Conversational Recommendations (RAG)"
    ]
)

top_n = st.sidebar.slider("Number of recommendations", 5, 20, 10)


if option == "Popular Movies":
    st.subheader("🔥 Popular Movies")
    st.info(
        "This method recommends movies with high average ratings, while filtering out "
        "movies with very few ratings."
    )

    min_ratings = st.slider("Minimum ratings", 1, 20, 5)

    result = recommend_popular_movies(
        top_n=top_n,
        min_ratings=min_ratings
    )

    display_recommendations(result, include_ratings=True)


elif option == "Content-Based Recommendation":
    st.subheader("🎯 Content-Based Recommendation")
    st.info(
        "Content-based filtering recommends movies similar to a selected movie using "
        "title, genre, and release year."
    )

    movie_title = st.selectbox("Choose a movie", movies["title"].tolist())

    result = recommend_content_based(
        movie_title=movie_title,
        top_n=top_n
    )

    display_recommendations(result)


elif option == "User-Based Personalized Recommendation":
    st.subheader("👤 Personalized User Recommendations")
    st.info(
        "Collaborative filtering recommends movies based on rating patterns from "
        "similar users and movies."
    )

    user_id = st.selectbox(
        "Choose user ID",
        sorted(ratings["user_id"].unique())
    )

    result = recommend_for_user(
        user_id=user_id,
        top_n=top_n
    )

    display_recommendations(result)


elif option == "Hybrid Recommendation":
    st.subheader("🧠 Hybrid Recommendation")

    st.info(
        "Hybrid recommendation combines content similarity and collaborative filtering "
        "to balance personalization and movie similarity."
    )

    # -----------------------------
    # Select user
    # -----------------------------
    user_id = st.selectbox(
        "Choose user ID",
        sorted(ratings["user_id"].unique())
    )

    # -----------------------------
    # Select movie
    # -----------------------------
    movie_title = st.selectbox(
        "Choose a movie",
        movies["title"].tolist()
    )

    # -----------------------------
    # Build GenRec-inspired user context
    # -----------------------------
    user_context = build_user_context(
        user_id=user_id,
        ratings_df=ratings,
        movies_df=movies,
        max_liked=5,
        max_disliked=5
    )

    with st.expander("🧠 User Context for LLM Reranking"):
        st.text(user_context)

    # -----------------------------
    # Hybrid weights
    # -----------------------------
    content_weight = st.slider(
        "Content-based weight",
        0.0,
        1.0,
        0.5
    )

    cf_weight = 1.0 - content_weight

    st.write(
        f"Collaborative filtering weight: **{cf_weight:.2f}**"
    )

    # -----------------------------
    # Generate hybrid recommendations
    # -----------------------------
    result = hybrid_recommend(
        user_id=user_id,
        movie_title=movie_title,
        top_n=top_n,
        content_weight=content_weight
    )

    # -----------------------------
    # Format candidates for LLM
    # -----------------------------
    candidate_context = format_candidates(result)

    with st.expander("🎬 Candidate Context for LLM Reranking"):
        st.text(candidate_context)

    # -----------------------------
    # Build LLM reranking prompt
    # -----------------------------
    reranking_prompt = build_reranking_prompt(
        user_context=user_context,
        candidate_context=candidate_context
    )

    with st.expander("📝 LLM Reranking Prompt"):
        st.text(reranking_prompt)

    # -----------------------------
    # Display hybrid recommendations
    # -----------------------------
    display_recommendations(result)


elif option == "Semantic Search (AI-Powered)":
    st.subheader("🤖 Semantic Search")
    st.info(
        "Semantic search uses a pretrained Sentence Transformer "
        "(`all-MiniLM-L6-v2`) to embed movie metadata and your query into "
        "the same vector space, then retrieves the closest matches using a "
        "**FAISS vector index** (exact nearest-neighbor search via inner "
        "product on normalized embeddings -- equivalent to cosine "
        "similarity, but using the same indexing primitive production "
        "retrieval/RAG systems use, instead of a brute-force scan). "
        "Unlike the TF-IDF content-based method, this can match on meaning "
        "rather than exact keywords -- e.g. try \"gritty crime thriller "
        "from the 90s\" or \"heartwarming animated family movie\"."
    )

    query = st.text_input(
        "Describe the kind of movie you're looking for",
        placeholder="e.g. gritty crime thriller from the 90s"
    )

    if query:
        result = faiss_search(
            query=query,
            movies_df=movies,
            index=movie_index,
            model=semantic_model,
            top_n=top_n
        )

        display_recommendations(result, include_similarity=True)
    else:
        st.write("Enter a description above to get AI-powered recommendations.")


elif option == "Conversational Recommendations (RAG)":
    st.subheader("💬 Conversational Recommendations (RAG)")
    st.info(
        "This is a Retrieval-Augmented Generation (RAG) pipeline: your "
        "message retrieves candidate movies from the FAISS vector index "
        "(same retrieval as Semantic Search), then an LLM (Anthropic's "
        "`claude-3-5-haiku` via Amazon Bedrock) "
        "recommends from *only* those retrieved candidates and explains "
        "why -- it can't invent movies outside the retrieved list. "
        "Try a follow-up like \"something funnier\" or \"more recent\" to "
        "refine within the conversation."
    )

    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []

    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant" and message.get("candidates") is not None:
                with st.expander("Retrieved candidates (grounding context)"):
                    display_recommendations(message["candidates"], include_similarity=True)

    user_message = st.chat_input("e.g. Show me a feel-good movie for a rainy day")

    if user_message:
        st.session_state.rag_messages.append({"role": "user", "content": user_message})
        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving candidates and generating a response..."):
                try:
                    response_text, candidates = rag_chat.chat(
                        conversation_history=st.session_state.rag_messages,
                        movies_df=movies,
                        index=movie_index,
                        embedding_model=semantic_model,
                        top_n=top_n,
                    )
                except RuntimeError as e:
                    response_text = str(e)
                    candidates = None

                st.write(response_text)
                if candidates is not None and not candidates.empty:
                    with st.expander("Retrieved candidates (grounding context)"):
                        display_recommendations(candidates, include_similarity=True)

        st.session_state.rag_messages.append({
            "role": "assistant",
            "content": response_text,
            "candidates": candidates,
        })

    if st.session_state.rag_messages:
        if st.button("Clear conversation"):
            st.session_state.rag_messages = []
            st.rerun()


st.markdown("---")
st.caption(
    "Built as part of an Applied AI / Machine Learning portfolio project."
)