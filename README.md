# CineMatch — Content-Based Movie Recommendation System

A content-based movie recommender built on the TMDB 5000-movies dataset, with
a cinematic Streamlit interface.

## Files

| File | Purpose |
|---|---|
| `movies.csv` | Raw dataset (your upload) |
| `recommender.py` | Core pipeline: cleaning, feature engineering, TF-IDF model, recommendation function |
| `preprocess.py` | Standalone script — runs the pipeline and prints an EDA summary, saves `movies_cleaned.csv` |
| `app.py` | Streamlit app (the interface) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

### 1. (Optional) Inspect the preprocessing

```bash
python preprocess.py
```
This prints missing-value counts, cleaning steps, vocabulary size, and the
top-rated movies, and saves `movies_cleaned.csv` for inspection.

### 2. Launch the app

```bash
streamlit run app.py
```
Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## How the recommender works

1. **Cleaning** — drops duplicate titles, fills missing genres/keywords/cast/
   overview/tagline with empty strings, imputes missing runtime with the
   median, parses `release_date` into a `release_year`, and discounts ratings
   backed by fewer than 5 votes.
2. **Feature engineering** — builds a combined `tags` text field per movie
   from genres (double-weighted), keywords, overview, tagline, cast, and
   director (triple-weighted so directorial style carries more signal), and
   computes an IMDB-style **Bayesian weighted rating** for fair "Top Rated"
   sorting (protects against a 10/10 with 2 votes outranking a 8.5/10 with
   50,000 votes).
3. **Modeling** — vectorizes the `tags` corpus with **TF-IDF**
   (`max_features=15000`, English stop words removed) and ranks candidates
   for a chosen movie by **cosine similarity**, computed on demand for just
   the query row (no need to store a full N×N matrix in memory).

## The interface

- **🎯 Get Recommendations** — pick any movie, see its poster/synopsis, and
  get a ranked grid of the most similar titles with a "match %" score.
- **🔍 Explore & Filter** — browse the whole catalog with genre, year, and
  rating filters.
- **🏆 Top Rated** — the 20 best movies by weighted score.
- **📈 Insights** — genre distribution, rating histogram, releases per year,
  and a budget-vs-revenue scatter plot.

Posters are placeholders by default. Paste a free
[TMDB API key](https://www.themoviedb.org/settings/api) into the sidebar to
show real poster art.
