import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Netflix-Style Movie Recommendation System")

movies = pd.read_csv("data/movies.csv")
users = pd.read_csv("data/users.csv")
ratings = pd.read_csv("data/ratings.csv")

# Content-based setup
movies["content"] = (
    movies["title"].fillna("") + " " +
    movies["genre"].fillna("") + " " +
    movies["release_year"].astype(str)
)

tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["content"])
content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Collaborative filtering setup
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

movie_similarity = cosine_similarity(user_movie_matrix.T)

movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index=user_movie_matrix.columns,
    columns=user_movie_matrix.columns
)


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

            scores[similar_movie_id] = scores.get(similar_movie_id, 0) + similarity_score * rating

    if not scores:
        return pd.DataFrame()

    recommended_ids = (
        pd.Series(scores)
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return movies[movies["movie_id"].isin(recommended_ids)]


def hybrid_recommend(user_id, movie_title, top_n=10, content_weight=0.5, cf_weight=0.5):
    matches = movies[movies["title"].str.lower().str.contains(movie_title.lower())]

    if matches.empty or user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    movie_idx = matches.index[0]
    movie_id = movies.loc[movie_idx, "movie_id"]

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

    recommended_ids = combined_scores.sort_values(ascending=False).head(top_n).index

    return movies[movies["movie_id"].isin(recommended_ids)]


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
    st.subheader("Popular Movies")
    min_ratings = st.slider("Minimum ratings", 1, 20, 5)
    result = recommend_popular_movies(top_n=top_n, min_ratings=min_ratings)
    st.dataframe(result)

elif option == "Content-Based Recommendation":
    st.subheader("Content-Based Recommendation")
    movie_title = st.selectbox("Choose a movie", movies["title"].tolist())

    result = recommend_content_based(movie_title, top_n=top_n)
    st.dataframe(result)

elif option == "User-Based Personalized Recommendation":
    st.subheader("Personalized User Recommendations")
    user_id = st.selectbox("Choose user ID", sorted(ratings["user_id"].unique()))

    result = recommend_for_user(user_id=user_id, top_n=top_n)
    st.dataframe(result)

elif option == "Hybrid Recommendation":
    st.subheader("Hybrid Recommendation")

    user_id = st.selectbox("Choose user ID", sorted(ratings["user_id"].unique()))
    movie_title = st.selectbox("Choose a movie", movies["title"].tolist())

    content_weight = st.slider("Content weight", 0.0, 1.0, 0.5)
    cf_weight = 1.0 - content_weight

    result = hybrid_recommend(
        user_id=user_id,
        movie_title=movie_title,
        top_n=top_n,
        content_weight=content_weight,
        cf_weight=cf_weight
    )

    st.dataframe(result)