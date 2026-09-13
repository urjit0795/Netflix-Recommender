import json

from google import genai
from google.genai import types


# We already verified that this model works with your API key.
GEMINI_MODEL = "gemini-3.6-flash"


def build_user_context(
    user_id,
    ratings_df,
    movies_df,
    max_liked=5,
    max_disliked=5,
):
    """
    Convert a user's structured rating history into natural-language
    context for the LLM reranker.

    Includes both highly rated and poorly rated movies so the LLM
    can understand positive and negative preferences.
    """

    user_ratings = ratings_df[
        ratings_df["user_id"] == user_id
    ].copy()

    if user_ratings.empty:
        return "No rating history available for this user."

    user_history = user_ratings.merge(
        movies_df[
            [
                "movie_id",
                "title",
                "genre",
                "release_year",
            ]
        ],
        on="movie_id",
        how="left",
    )

    # Highest-rated movies
    liked_movies = (
        user_history
        .sort_values("rating", ascending=False)
        .head(max_liked)
    )

    # Lowest-rated movies
    disliked_movies = (
        user_history
        .sort_values("rating", ascending=True)
        .head(max_disliked)
    )

    context_lines = [
        "USER PREFERENCE HISTORY",
        "",
        "Movies the user liked:",
    ]

    for _, row in liked_movies.iterrows():
        context_lines.append(
            f"- {row['title']} "
            f"({row['release_year']}) | "
            f"Genre: {row['genre']} | "
            f"Rating: {row['rating']}/5"
        )

    context_lines.extend(
        [
            "",
            "Movies the user disliked:",
        ]
    )

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
    Convert recommendation candidates into compact natural-language
    context for the LLM reranker.

    Expected columns:
        movie_id
        title
        genre
        release_year

    Optional column:
        hybrid_score
    """

    if candidates_df is None or candidates_df.empty:
        return "No candidate movies available."

    context_lines = [
        "CANDIDATE MOVIES",
        "",
    ]

    for rank, (_, row) in enumerate(
        candidates_df.iterrows(),
        start=1,
    ):
        hybrid_score = row.get("hybrid_score")

        if hybrid_score is not None:
            try:
                hybrid_score = f"{float(hybrid_score):.4f}"
            except (TypeError, ValueError):
                hybrid_score = str(hybrid_score)
        else:
            hybrid_score = "N/A"

        context_lines.append(
            f"{rank}. Movie ID: {row['movie_id']} | "
            f"{row['title']} ({row['release_year']})"
        )

        context_lines.append(
            f"   Genre: {row['genre']}"
        )

        context_lines.append(
            f"   Hybrid score: {hybrid_score}"
        )

        context_lines.append("")

    return "\n".join(context_lines).strip()


def build_reranking_prompt(
    user_context,
    candidate_context,
):
    """
    Combine user preference history and recommendation candidates
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

The candidate movies were already retrieved by a recommendation
system using traditional recommendation and/or semantic retrieval.

The existing recommendation score is useful evidence, but you are
allowed to change the ranking if the user's preference history
suggests a better ordering.

{user_context}

{candidate_context}

Return the reranked movies as JSON.

Use exactly this structure:

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
- Use the exact movie_id values provided in the candidate list.
- Rank every candidate exactly once.
- Do not duplicate candidates.
- Rank 1 is the strongest recommendation.
- score must be between 0 and 1.
- Keep each reason concise.
- Return valid JSON only.
"""

    return prompt.strip()


def get_gemini_client():
    """
    Create a Gemini client.

    The SDK automatically reads GEMINI_API_KEY
    from the environment.
    """

    return genai.Client()


def call_llm_reranker(
    prompt,
    client=None,
):
    """
    Send the reranking prompt to Google Gemini.

    Returns
    -------
    str
        Raw JSON text returned by Gemini.
    """

    if client is None:
        client = get_gemini_client()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=4000,
        ),
    )

    if not response.text:
        raise ValueError(
            "Gemini returned an empty response."
        )

    return response.text.strip()


def validate_reranked_results(
    reranked_results,
    candidates_df,
):
    """
    Validate that the LLM returned exactly the candidate movies
    supplied to it.

    Checks:
    - response is a list
    - every item has required fields
    - no candidate IDs were invented
    - no candidates were omitted
    - no candidates were duplicated
    - rank values are valid
    - scores are between 0 and 1
    """

    if not isinstance(reranked_results, list):
        raise ValueError(
            "LLM reranking response must be a JSON list."
        )

    required_fields = {
        "movie_id",
        "rank",
        "score",
        "reason",
    }

    candidate_ids = set(
        candidates_df["movie_id"].tolist()
    )

    returned_ids = []

    for result in reranked_results:
        if not isinstance(result, dict):
            raise ValueError(
                "Every reranking result must be a JSON object."
            )

        missing_fields = required_fields - set(result.keys())

        if missing_fields:
            raise ValueError(
                f"Reranking result is missing fields: "
                f"{missing_fields}"
            )

        movie_id = result["movie_id"]

        # Normalize numeric IDs in case Gemini returns them
        # as strings instead of integers.
        try:
            movie_id = int(movie_id)
            result["movie_id"] = movie_id
        except (TypeError, ValueError):
            pass

        if movie_id not in candidate_ids:
            raise ValueError(
                f"LLM returned movie_id {movie_id}, "
                "which was not in the candidate set."
            )

        returned_ids.append(movie_id)

        try:
            score = float(result["score"])
            result["score"] = score
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid score for movie_id {movie_id}: "
                f"{result['score']}"
            ) from exc

        if not 0 <= score <= 1:
            raise ValueError(
                f"Score for movie_id {movie_id} "
                "must be between 0 and 1."
            )

        try:
            result["rank"] = int(result["rank"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rank for movie_id {movie_id}: "
                f"{result['rank']}"
            ) from exc

    if len(returned_ids) != len(set(returned_ids)):
        raise ValueError(
            "LLM returned duplicate movie IDs."
        )

    returned_id_set = set(returned_ids)

    if returned_id_set != candidate_ids:
        missing_ids = candidate_ids - returned_id_set

        raise ValueError(
            f"LLM omitted candidate movie IDs: "
            f"{sorted(missing_ids)}"
        )

    expected_ranks = set(
        range(1, len(candidates_df) + 1)
    )

    returned_ranks = {
        result["rank"]
        for result in reranked_results
    }

    if returned_ranks != expected_ranks:
        raise ValueError(
            "LLM returned invalid ranking values. "
            f"Expected ranks: {sorted(expected_ranks)}"
        )

    return True


def rerank_candidates(
    user_id,
    ratings_df,
    movies_df,
    candidates_df,
    client=None,
):
    """
    Run the complete LLM reranking pipeline.

    Steps:
    1. Build user preference context.
    2. Format recommendation candidates.
    3. Build reranking prompt.
    4. Call Google Gemini.
    5. Parse the JSON response.
    6. Validate the returned ranking.

    Returns
    -------
    list[dict]
        Reranked candidate movies.
    """

    if candidates_df is None or candidates_df.empty:
        return []

    user_context = build_user_context(
        user_id=user_id,
        ratings_df=ratings_df,
        movies_df=movies_df,
    )

    candidate_context = format_candidates(
        candidates_df
    )

    prompt = build_reranking_prompt(
        user_context=user_context,
        candidate_context=candidate_context,
    )

    raw_response = call_llm_reranker(
        prompt=prompt,
        client=client,
    )

    try:
        reranked_results = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Raw response:\n{raw_response}"
        ) from exc

    validate_reranked_results(
        reranked_results=reranked_results,
        candidates_df=candidates_df,
    )

    reranked_results = sorted(
        reranked_results,
        key=lambda x: int(x["rank"]),
    )

    return reranked_results