# 🎬 Netflix Hybrid Recommendation System

An end-to-end movie recommendation system inspired by Netflix, built using multiple recommendation techniques including popularity-based filtering, content-based filtering, collaborative filtering, personalized recommendations, and hybrid recommendation systems.

The project also includes evaluation metrics and an interactive Streamlit application for exploring recommendations.

---

# 🚀 Features

## ✅ Popularity-Based Recommendations
Recommend top-rated and most popular movies based on user ratings.

## ✅ Content-Based Filtering
Recommend movies similar to a selected movie using:
- TF-IDF vectorization
- Cosine similarity
- Movie metadata such as genre and title

## ✅ Collaborative Filtering
Recommend movies using:
- User-item interaction matrix
- Item-item similarity
- Cosine similarity

## ✅ Personalized Recommendations
Generate recommendations tailored to a specific user based on historical ratings.

## ✅ Hybrid Recommendation System
Combine:
- Content-based filtering
- Collaborative filtering

to generate stronger recommendations.

## ✅ Evaluation Metrics
Implemented:
- Precision@K
- Recall@K

to measure recommendation quality.

## ✅ Interactive Streamlit App
Users can:
- Search for movies
- Get similar movie recommendations
- Get personalized recommendations
- Use hybrid recommendations
- Explore popular movies

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

# 📂 Project Structure

```text
NETFLIX_RECOMMENDATION/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── movies.csv
│   ├── users.csv
│   └── ratings.csv
│
├── notebooks/
│   └── netflix_recommendation.ipynb
│
├── recommender/
│   ├── engine.py
│   └── evaluation.py
│
└── outputs/