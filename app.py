import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Recommendation System")
st.write(
    "An interactive app to explore movie recommendations using popularity-based, "
    "content-based, collaborative filtering, and hybrid recommendation methods."
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

    matches = movies[movies["title"].str.lower().str.contains(movie_title.lower())]

    if matches.empty or user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    movie_idx = matches.index[0]
    movie_id = movies.loc[movie_idx, "movie_id"]

    if movie_id not in movie_similarity_df.columns:
        return pd.DataFrame()

    content_scores = pd.Series(
        content_similarity[movie_idx],
        index=movies["movie_id"]
    )

    cf_scores = movie_similarity_df[movie_id]

    user_ratings = user_movie_matrix.loc[user_id]
    watched_movies = user_ratings[user_ratings > 0].index.tolist()

    combined_scores = (
        content_weight * content_scores +
        cf_weight * cf_scores
    )

    combined_scores = combined_scores.drop(labels=watched_movies, errors="ignore")

    recommended_ids = (
        combined_scores
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return movies[movies["movie_id"].isin(recommended_ids)]


def display_recommendations(result, include_ratings=False):
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
    else:
        display_cols = ["title", "genre", "release_year"]

    available_cols = [col for col in display_cols if col in result.columns]

    st.dataframe(
        result[available_cols],
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
        "Hybrid Recommendation"
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

    user_id = st.selectbox(
        "Choose user ID",
        sorted(ratings["user_id"].unique())
    )

    movie_title = st.selectbox("Choose a movie", movies["title"].tolist())

    content_weight = st.slider(
        "Content-based weight",
        0.0,
        1.0,
        0.5
    )

    cf_weight = 1.0 - content_weight

    st.write(f"Collaborative filtering weight: **{cf_weight:.2f}**")

    result = hybrid_recommend(
        user_id=user_id,
        movie_title=movie_title,
        top_n=top_n,
        content_weight=content_weight
    )

    display_recommendations(result)


st.markdown("---")
st.caption(
    "Built as part of an Applied AI / Machine Learning portfolio project."
)