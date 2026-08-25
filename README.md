<div align="center">

# 🎬 Netflix AI Recommendation Platform

### From Classical Recommendation Systems to LLM-Powered Recommendations

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue)
![AWS](https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazonaws)

</div>

---

## 🚀 Overview

An end-to-end movie recommendation platform exploring the evolution from **traditional recommendation algorithms to modern LLM-powered recommendation systems**.

The project combines:

* Popularity, content-based and collaborative filtering
* Hybrid recommendation
* Sentence Transformer embeddings
* FAISS semantic retrieval
* Retrieval-Augmented Generation (RAG)
* Claude via AWS Bedrock
* GenRec-inspired context engineering and LLM reranking

The goal is not to reproduce Netflix's production system, but to experiment with the architecture and techniques behind modern recommendation platforms.

---

## 🧠 How It Works

```text
                    User
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
   Rating History           Movie / Query
          │                       │
          ▼                       ▼
  User Preference          Candidate Generation
      Context               ├─ Hybrid Recommender
          │                 └─ FAISS Semantic Search
          │                       │
          └───────────┬───────────┘
                      ▼
                 LLM Layer
                Claude / Bedrock
                      │
             ┌────────┴────────┐
             ▼                 ▼
      Conversational       LLM Reranking
      Recommendations          🚧
```

---

## ✨ Features

| Feature                             | Status |
| ----------------------------------- | :----: |
| Popularity Recommendations          |    ✅   |
| Content-Based Filtering             |    ✅   |
| Collaborative Filtering             |    ✅   |
| Hybrid Recommendations              |    ✅   |
| Precision@K / Recall@K              |    ✅   |
| Sentence Transformer Embeddings     |    ✅   |
| FAISS Semantic Search               |    ✅   |
| Conversational RAG                  |    ✅   |
| AWS Bedrock / Claude Integration    |    ✅   |
| GenRec-Inspired Context Engineering |    ✅   |
| LLM Reranking                       |   🚧   |
| Ranking / LLM Evaluation            |    ⏳   |
| Recommendation Agent                |    ⏳   |

---

## 🔍 Semantic Search + RAG

The platform supports natural-language movie discovery using **Sentence Transformers and FAISS**.

For example:

```text
"gritty crime thriller from the 90s"
```

The query is embedded and matched against the movie catalog using semantic similarity.

For conversational recommendations, retrieved movies are passed to **Claude through AWS Bedrock**, allowing the LLM to recommend and explain movies while remaining grounded in retrieved candidates.

```text
User Query
    ↓
Sentence Transformer
    ↓
FAISS Retrieval
    ↓
Candidate Movies
    ↓
Claude / AWS Bedrock
    ↓
Grounded Recommendation
```

---

## 🧠 GenRec-Inspired LLM Reranking

The latest experiment is inspired by Netflix's **GenRec** research and explores moving part of the recommendation problem from **feature engineering toward context engineering**.

Instead of replacing the existing recommender, the hybrid model remains responsible for candidate generation.

```text
User Rating History
        ↓
Natural-Language Preference Context
        │
        ├──────────────┐
        │              │
Hybrid Recommender     │
        ↓              │
Candidate Movies       │
        │              │
        └──────┬───────┘
               ▼
        Reranking Prompt
               ↓
          LLM Reranker 🚧
```

The user context includes both positive and negative preference signals, while candidate context includes movie metadata and the original hybrid recommendation score.

The next step is to use Claude to rerank these candidates and compare the result against the original hybrid ranking.

---

## 📊 Evaluation

Currently implemented:

* **Precision@K**
* **Recall@K**

Planned for the reranking experiment:

* **NDCG@K**
* Hybrid vs. LLM ranking comparison
* Recommendation relevance
* Latency and cost

---

## 🛠 Tech Stack

| Area             | Technologies          |
| ---------------- | --------------------- |
| Language         | Python                |
| Machine Learning | Scikit-learn          |
| Data             | Pandas, NumPy         |
| Embeddings       | Sentence Transformers |
| Vector Search    | FAISS                 |
| LLM              | Claude                |
| Cloud AI         | AWS Bedrock           |
| UI               | Streamlit             |

---

## 📂 Project Structure

```text
Netflix-Recommender/
│
├── app.py
├── semantic_search.py
├── vector_search.py
├── rag_chat.py
├── llm_reranker.py
├── requirements.txt
├── netflix_recommender.ipynb
│
├── data/
├── diagrams/
└── docs/
```

---

## 🗺 Roadmap

```text
Classical Recommendations      ✅
          ↓
Hybrid Recommendation          ✅
          ↓
Semantic Search + FAISS        ✅
          ↓
Conversational RAG             ✅
          ↓
GenRec Context Engineering     ✅
          ↓
LLM Reranking                  🚧
          ↓
Ranking / LLM Evaluation       ⏳
          ↓
Recommendation Agent           ⏳
```

---

## 🎯 What I'm Exploring

This project focuses on practical questions in modern recommendation systems:

* Where should traditional ML end and LLMs begin?
* Can semantic retrieval improve recommendation discovery?
* Can richer user context improve ranking?
* How should LLM-powered recommendations be evaluated?
* How can recommendations remain grounded and explainable?

---

## ⭐ About

Built as an **Applied AI / Machine Learning portfolio project** exploring the progression from traditional recommendation algorithms to retrieval, RAG, and LLM-powered ranking.
