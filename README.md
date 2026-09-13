# 🎬 Netflix AI Recommendation Platform

An end-to-end recommendation system exploring the evolution from **classical recommendation algorithms → semantic retrieval → RAG → LLM-based reranking**.

The goal of this project is to understand how traditional recommendation systems and modern LLM-based approaches can work together rather than replacing one another.

---

## 🏗️ Architecture

```text
User
 │
 ├── Rating History
 │       ↓
 │   User Preference Context
 │
 └── Natural-Language Query
         ↓
   Sentence Transformers
         ↓
       FAISS
         ↓
   Candidate Movies
         ↓
   Context Engineering
         ↓
    LLM Reranker ✅
         ↓
 Final Recommendations
```

The architecture follows a simple idea:

### **Retrieve → Reason → Rank**

**Retrieve:** Traditional recommendation models and semantic search identify relevant candidate movies.

**Reason:** User history and candidate metadata are converted into structured context for the LLM.

**Rank:** The LLM evaluates a constrained candidate set and produces a final ranking.

> **FAISS retrieves. The LLM reasons.**

---

## ✨ Current Features

| Feature | Status |
| --- | --- |
| Popularity-Based Recommendations | ✅ |
| Content-Based Filtering | ✅ |
| Collaborative Filtering | ✅ |
| Hybrid Recommendations | ✅ |
| Precision@K / Recall@K | ✅ |
| Sentence Transformer Embeddings | ✅ |
| Semantic Search | ✅ |
| FAISS Vector Search | ✅ |
| Conversational RAG | ✅ |
| GenRec-Inspired Context Engineering | ✅ |
| Gemini LLM Reranker | ✅ |
| LLM / Ranking Evaluation | ⏳ |
| Recommendation Agent | ⏳ |

---

## 1️⃣ Classical Recommendation Systems

The project started by implementing several recommendation baselines:

### Popularity-Based

Ranks movies using overall engagement and rating signals.

### Content-Based Filtering

Uses movie metadata and TF-IDF similarity to identify related movies.

### Collaborative Filtering

Uses user-rating behavior to identify recommendations based on similar preferences.

### Hybrid Recommendations

Combines multiple recommendation signals into a unified candidate ranking.

These models provide the baseline against which newer AI-based approaches can be evaluated.

---

## 2️⃣ Semantic Search

Keyword matching can struggle when the user's intent doesn't exactly match movie metadata.

Semantic search allows queries such as:

```text
"real-world true story documentary"
```

The system:

```text
Query
  ↓
Sentence Transformer
  ↓
Query Embedding
  ↓
FAISS Vector Search
  ↓
Semantically Similar Movies
```

Movie metadata is embedded using a pretrained Sentence Transformer and stored for efficient retrieval.

FAISS provides fast similarity search across those embeddings.

---

## 3️⃣ Conversational RAG

Semantic retrieval is extended into a conversational recommendation workflow.

```text
User Query
    ↓
Semantic Retrieval
    ↓
FAISS
    ↓
Candidate Movies
    ↓
LLM
    ↓
Grounded Recommendation
```

Instead of asking the LLM to generate arbitrary movie recommendations, the model receives retrieved candidates as context.

This keeps the response grounded in the recommendation catalog while allowing natural-language interaction.

---

## 4️⃣ GenRec-Inspired LLM Reranking

The latest stage of the project uses an **LLM as a contextual reranker rather than as the primary retrieval system**.

The approach is inspired by Netflix's **GenRec** work on generative recommendation ranking.

```text
User Rating History
        ↓
Preference Context
        │
        │
Hybrid / Semantic Retrieval
        ↓
Candidate Movies
        │
        └──────────────┐
                       ↓
                Reranking Context
                       ↓
                  Gemini LLM
                    Reranker
                       ↓
                  Final Ranking
```

The reranker receives two important pieces of context:

**User context**
- Movies the user rated highly
- Movies the user disliked
- Genre and preference signals

**Candidate context**
- Candidate movie metadata
- Existing recommendation scores
- Relevant retrieval information

The LLM then reranks the retrieved candidates and returns:

- Final rank
- Relevance score
- Short explanation for the ranking decision

This allows the LLM to reason over a **small, high-quality candidate set** rather than the entire catalog.

---

## 🧠 Why Retrieval + LLM Reranking?

LLMs are powerful reasoning systems, but they are not necessarily the most efficient retrieval engines.

A multi-stage recommendation architecture lets each component do what it does best:

```text
Vector Search → Fast Candidate Retrieval

Traditional ML → Recommendation Signals

LLM → Contextual Reasoning + Reranking
```

This creates a hybrid system where traditional recommendation techniques and generative AI complement each other.

Importantly, the LLM does not replace the existing recommendation system. It operates as a downstream ranking layer over candidates already identified by retrieval and recommendation models.

---

## 📊 Evaluation

Currently implemented:

- Precision@K
- Recall@K

These metrics are currently used to evaluate the traditional recommendation pipeline.

The next phase will compare:

```text
Baseline Recommendations
        vs.
Hybrid Recommendations
        vs.
LLM-Reranked Recommendations
```

Planned evaluation includes:

- NDCG@K
- Ranking quality
- LLM-as-a-Judge relevance
- Latency
- LLM inference cost

The goal is not simply to add an LLM, but to determine whether it **measurably improves recommendation quality** and under what conditions the additional inference cost is justified.

---

## 🛠️ Tech Stack

| Area | Technologies |
| --- | --- |
| Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Generative AI | Google Gemini, RAG, LLM Reranking |
| Application | Streamlit |

---

## 🗺️ Project Roadmap

```text
Classical Recommendations     ✅
          ↓
Hybrid Recommendation         ✅
          ↓
Semantic Search               ✅
          ↓
FAISS Vector Retrieval        ✅
          ↓
Conversational RAG            ✅
          ↓
GenRec Context Engineering    ✅
          ↓
Gemini LLM Reranking          ✅
          ↓
Ranking + LLM Evaluation      ⏳
          ↓
Recommendation Agent          ⏳
```

---

## 🔬 What I'm Exploring

This project is ultimately about answering a few practical Applied AI questions:

- Where should traditional ML end and LLMs begin?
- Should an LLM retrieve, reason, rank — or some combination of the three?
- How much does user context affect LLM ranking?
- Does LLM reranking actually outperform traditional recommendation methods?
- Under what conditions does LLM reranking provide the most value?
- How should LLM-powered recommendations be evaluated?
- How do recommendation quality, latency, and inference cost trade off?

---

## 🚀 Running the Project

Clone the repository:

```bash
git clone https://github.com/urjit0795/Netflix-Recommender.git
cd Netflix-Recommender
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable.

**Windows PowerShell:**

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Then run the Streamlit application:

```bash
streamlit run app.py
```

> Never commit API keys or credentials to the repository.

---

## 📚 Inspiration

The LLM reranking portion of this project is inspired by Netflix's **GenRec** research, which explores using generative models as recommendation rankers.

Rather than reproducing Netflix's production system, this project experiments with the underlying idea at a smaller scale:

> **Retrieve strong candidates first, then use an LLM to reason about their final ranking.**

The next stage of the project will evaluate whether this additional reasoning layer actually produces better rankings and whether the improvement justifies the additional latency and inference cost.

---

## 👤 Author

**Urjit Kurulkar**

Senior Data Scientist focused on production ML, recommendation systems, Generative AI, RAG, and Applied AI engineering.