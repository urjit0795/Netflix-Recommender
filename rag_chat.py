"""
RAG (Retrieval-Augmented Generation) conversational recommendations.

Given a free-text, conversational query (e.g. "show me something like that
but funnier"), this module:

  1. Retrieves the most relevant movies from the FAISS vector index
     (vector_search.faiss_search) -- the "R" in RAG.
  2. Passes those retrieved movies as grounding context to an LLM
     (Anthropic's Claude, via Amazon Bedrock), which generates a
     natural-language recommendation/explanation -- the "G" in RAG.

This is deliberately a small, explicit RAG pipeline: no vector database
service, no agent framework -- just retrieval -> context -> generation,
the core pattern underneath more complex RAG systems.

Uses Amazon Bedrock for model access, authenticated via your normal AWS
credential chain (environment variables, ~/.aws/credentials, SSO, or an IAM
role) -- no separate Anthropic API key is needed. Set AWS_REGION (or
AWS_DEFAULT_REGION) to whichever region has Bedrock + Claude model access
enabled for your account.
"""

import os

import pandas as pd

from vector_search import faiss_search

# Bedrock model IDs differ from the direct Anthropic API's model names.
# This is the Bedrock model ID for Claude 3.5 Haiku -- confirm this matches
# the model ID enabled for your account/region in the Bedrock console
# (Model access page) if you hit a "model not found" / access-denied error.
CHAT_MODEL = "anthropic.claude-3-5-haiku-20241022-v1:0"
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable movie recommendation assistant for a "
    "streaming platform. You are given a shortlist of candidate movies "
    "retrieved from the catalog (with title, genre, release year, and a "
    "similarity score) and the user's request, including any earlier turns "
    "of the conversation. "
    "Recommend 1-3 movies from ONLY the candidates provided -- never invent "
    "movies that aren't in the candidate list. "
    "Briefly explain *why* each pick fits the user's request. "
    "If none of the candidates are a good fit, say so honestly instead of "
    "forcing a recommendation. Keep the response conversational and concise."
)


def _format_candidates(candidates: pd.DataFrame) -> str:
    """Render retrieved candidate movies as plain text for the LLM prompt."""
    lines = []
    for _, row in candidates.iterrows():
        lines.append(
            f"- {row['title']} ({row.get('genre', 'Unknown genre')}, "
            f"{row.get('release_year', 'unknown year')}) "
            f"[similarity: {row.get('similarity', 0):.3f}]"
        )
    return "\n".join(lines)


def get_anthropic_client():
    """Lazily import and construct the Anthropic-on-Bedrock client.

    Authenticates via the standard AWS credential chain (env vars,
    ~/.aws/credentials, SSO, or an IAM role) rather than an API key. Raises
    a clear error up front if no AWS region is configured, rather than
    failing deep inside the API call with a less obvious error.
    """
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        raise RuntimeError(
            "No AWS region configured. Set AWS_REGION (or AWS_DEFAULT_REGION) "
            "to the region where Bedrock + Claude model access is enabled, e.g.:\n"
            "  PowerShell: $env:AWS_REGION=\"us-east-1\"\n"
            "  bash:       export AWS_REGION=\"us-east-1\"\n"
            "AWS credentials themselves are picked up from your normal AWS "
            "credential chain (aws configure / SSO / IAM role) -- no separate "
            "API key is needed for Bedrock."
        )

    from anthropic import AnthropicBedrock

    return AnthropicBedrock(aws_region=region)


def retrieve_candidates(
    query: str,
    movies_df: pd.DataFrame,
    index,
    model,
    top_n: int = 8,
) -> pd.DataFrame:
    """Retrieval step: fetch the top_n most relevant movies for this query."""
    return faiss_search(
        query=query,
        movies_df=movies_df,
        index=index,
        model=model,
        top_n=top_n,
    )


def generate_response(
    conversation_history: list,
    candidates: pd.DataFrame,
    client=None,
) -> str:
    """
    Generation step: ask the LLM to recommend from the retrieved candidates,
    grounded in the conversation so far.

    conversation_history is a list of {"role": "user"|"assistant", "content": str}
    dicts, in chronological order (the latest user message last).
    """
    if client is None:
        client = get_anthropic_client()

    candidate_context = _format_candidates(candidates)

    # Only forward role/content to the API -- conversation_history entries
    # from the UI layer may carry extra bookkeeping fields (e.g. the
    # retrieved candidates DataFrame attached to assistant turns for
    # display), which don't belong in the API payload.
    clean_history = [
        {"role": m["role"], "content": m["content"]} for m in conversation_history
    ]

    # Anthropic's API (direct or via Bedrock) takes the system prompt as a
    # separate top-level parameter, not as a message in the messages list
    # (unlike OpenAI).
    messages = list(clean_history[:-1])  # prior turns, if any

    latest_user_message = clean_history[-1]["content"]
    messages.append({
        "role": "user",
        "content": (
            f"Candidate movies retrieved for this request:\n{candidate_context}\n\n"
            f"User's request: {latest_user_message}"
        ),
    })

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        temperature=0.4,
    )

    return response.content[0].text


def chat(
    conversation_history: list,
    movies_df: pd.DataFrame,
    index,
    embedding_model,
    client=None,
    top_n: int = 8,
) -> tuple:
    """
    Run one full RAG turn: retrieve candidates for the latest user message,
    then generate a grounded response.

    Returns (response_text, candidates_dataframe) so the UI can show both
    the generated answer and the underlying retrieved movies for transparency.
    """
    latest_user_message = conversation_history[-1]["content"]

    candidates = retrieve_candidates(
        query=latest_user_message,
        movies_df=movies_df,
        index=index,
        model=embedding_model,
        top_n=top_n,
    )

    if candidates.empty:
        return (
            "I couldn't find any relevant movies for that request in the catalog.",
            candidates,
        )

    response_text = generate_response(conversation_history, candidates, client=client)
    return response_text, candidates
