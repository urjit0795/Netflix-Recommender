def build_user_context(
    user_id,
    ratings_df,
    movies_df,
    max_liked=5,
    max_disliked=5
):
    """
    Convert a user's structured rating history into natural-language
    context that can later be passed to the LLM reranker.

    Includes both highly rated and poorly rated movies so the LLM
    can understand positive and negative preferences.
    """

    # Get ratings for the selected user
    user_ratings = ratings_df[
        ratings_df["user_id"] == user_id
    ].copy()

    if user_ratings.empty:
        return "No rating history available for this user."

    # Join ratings with movie metadata
    user_history = user_ratings.merge(
        movies_df[
            [
                "movie_id",
                "title",
                "genre",
                "release_year"
            ]
        ],
        on="movie_id",
        how="left"
    )

    # -----------------------------
    # Movies the user liked
    # -----------------------------
    liked_movies = (
        user_history
        .sort_values("rating", ascending=False)
        .head(max_liked)
    )

    # -----------------------------
    # Movies the user disliked
    # -----------------------------
    disliked_movies = (
        user_history
        .sort_values("rating", ascending=True)
        .head(max_disliked)
    )

    context_lines = [
        "USER PREFERENCE HISTORY",
        "",
        "Movies the user liked:"
    ]

    for _, row in liked_movies.iterrows():
        context_lines.append(
            f"- {row['title']} "
            f"({row['release_year']}) | "
            f"Genre: {row['genre']} | "
            f"Rating: {row['rating']}/5"
        )

    context_lines.extend([
        "",
        "Movies the user disliked:"
    ])

    for _, row in disliked_movies.iterrows():
        context_lines.append(
            f"- {row['title']} "
            f"({row['release_year']}) | "
            f"Genre: {row['genre']} | "
            f"Rating: {row['rating']}/5"
        )

    return "\n".join(context_lines)

def format_candidates(candidates_df):
    """
    Convert hybrid recommendation candidates into compact natural-language
    context for the LLM reranker.

    Expected columns:
        movie_id, title, genre, release_year, hybrid_score

    Returns
    -------
    str
        Formatted candidate list for inclusion in the reranking prompt.
    """

    if candidates_df is None or candidates_df.empty:
        return "No candidate movies available."

    context_lines = [
        "CANDIDATE MOVIES",
        ""
    ]

    for rank, (_, row) in enumerate(candidates_df.iterrows(), start=1):
        hybrid_score = row.get("hybrid_score")

        if hybrid_score is not None:
            try:
                hybrid_score = f"{float(hybrid_score):.4f}"
            except (TypeError, ValueError):
                hybrid_score = str(hybrid_score)
        else:
            hybrid_score = "N/A"

        context_lines.append(
            f"{rank}. {row['title']} ({row['release_year']})"
        )
        context_lines.append(
            f"   Genre: {row['genre']}"
        )
        context_lines.append(
            f"   Hybrid score: {hybrid_score}"
        )
        context_lines.append("")

    return "\n".join(context_lines).strip()

def build_reranking_prompt(user_context, candidate_context):
    """
    Combine user preference history and hybrid candidate movies
    into a prompt for the LLM reranker.
    """

    prompt = f"""
You are an expert movie recommendation ranking system.

Your task is to rerank a list of candidate movies for a user.

Use the user's preference history to understand:
- genres they prefer
- genres they dislike
- patterns in highly rated movies
- patterns in poorly rated movies
- release-year preferences
- broader thematic preferences

The candidate movies were already retrieved by a hybrid
recommendation system using content-based filtering and
collaborative filtering.

The hybrid score is useful evidence, but you are allowed to
change the ranking if the user's preference history suggests
a better ordering.

{user_context}

{candidate_context}

Return the reranked movies in JSON format only.

Use this structure:

[
    {{
        "movie_id": 123,
        "rank": 1,
        "score": 0.95,
        "reason": "Short explanation of why this movie fits the user."
    }}
]

Requirements:
- Only rank movies from the candidate list.
- Do not invent new movies.
- Rank every candidate exactly once.
- Rank 1 is the strongest recommendation.
- score must be between 0 and 1.
- Keep each reason concise.
- Return valid JSON only.
"""

    return prompt.strip()