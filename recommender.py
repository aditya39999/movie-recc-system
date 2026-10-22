"""
recommender.py
================
Core data-preprocessing + content-based recommendation engine, shared by
both the standalone preprocessing script (preprocess.py) and the Streamlit
app (app.py).

Pipeline:
  1. Load raw CSV, drop unusable rows, handle missing values
  2. Engineer features (release_year, genre_list, Bayesian weighted_score)
  3. Build a combined "tags" text corpus (genres + keywords + overview +
     tagline + cast + director, with genre/director up-weighted)
  4. Vectorize the corpus with TF-IDF and compute cosine similarity
  5. Expose get_recommendations() for querying similar movies
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

GENRE_WEIGHT = 2
DIRECTOR_WEIGHT = 3


def clean_text(text) -> str:
    """Lowercase, strip non-alphanumerics, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


COMPOUND_GENRE_FIXES = {
    ("Science", "Fiction"): "Science Fiction",
    ("TV", "Movie"): "TV Movie",
}


def _rebuild_genre_tokens(tokens):
    """Re-merge compound genre names that got split on whitespace."""
    result, i = [], 0
    while i < len(tokens):
        pair = (tokens[i], tokens[i + 1]) if i + 1 < len(tokens) else None
        if pair in COMPOUND_GENRE_FIXES:
            result.append(COMPOUND_GENRE_FIXES[pair])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def load_and_clean(raw_path: str) -> pd.DataFrame:
    """Load the raw movies CSV and return a cleaned, feature-engineered DataFrame."""
    df = pd.read_csv(raw_path)

    # ---- Drop unusable / duplicate rows ----
    df = df[df["title"].notna()].copy()
    df = df.drop_duplicates(subset=["title", "release_date"]).copy()

    # ---- Fill missing text fields ----
    text_cols = ["genres", "keywords", "overview", "tagline", "cast",
                 "director", "homepage", "production_companies",
                 "production_countries", "spoken_languages"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # ---- Numeric / date cleanup ----
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    median_year = int(df["release_date"].dt.year.median())
    df["release_year"] = df["release_date"].dt.year.fillna(median_year).astype(int)

    # Ratings based on very few votes aren't trustworthy for display
    df["vote_average_display"] = df["vote_average"].where(df["vote_count"] >= 5, np.nan)
    df["vote_average_display"] = df["vote_average_display"].fillna(df["vote_average"])

    # ---- Genre list + display string ----
    df["genre_list"] = df["genres"].apply(
        lambda g: _rebuild_genre_tokens([t for t in g.split() if t]) if isinstance(g, str) else []
    )
    df["genres_display"] = df["genre_list"].apply(lambda lst: ", ".join(lst) if lst else "Unknown")

    # ---- Bayesian weighted rating (like IMDB's Top 250 formula) ----
    C = df["vote_average"].mean()
    m = df["vote_count"].quantile(0.75)
    df["weighted_score"] = (
        df["vote_count"] / (df["vote_count"] + m) * df["vote_average"]
        + m / (df["vote_count"] + m) * C
    )

    # ---- Combined tag corpus for content-based similarity ----
    genres_clean = df["genres"].apply(clean_text)
    keywords_clean = df["keywords"].apply(clean_text)
    overview_clean = df["overview"].apply(clean_text)
    tagline_clean = df["tagline"].apply(clean_text)
    cast_clean = df["cast"].apply(clean_text)
    director_clean = df["director"].apply(clean_text)

    df["tags"] = (
        (genres_clean + " ") * GENRE_WEIGHT
        + keywords_clean + " "
        + overview_clean + " "
        + tagline_clean + " "
        + cast_clean + " "
        + ((director_clean + " ") * DIRECTOR_WEIGHT)
    )

    keep_cols = [
        "id", "title", "original_title", "genres_display", "genre_list",
        "overview", "tagline", "director", "cast", "release_date",
        "release_year", "runtime", "vote_average", "vote_average_display",
        "vote_count", "weighted_score", "popularity", "budget", "revenue",
        "original_language", "status", "homepage", "tags",
    ]
    return df[keep_cols].reset_index(drop=True)


def build_similarity_model(df: pd.DataFrame):
    """Fit TF-IDF over the tags corpus and return (vectorizer, tfidf_matrix)."""
    vectorizer = TfidfVectorizer(max_features=15000, stop_words="english", min_df=1)
    tfidf_matrix = vectorizer.fit_transform(df["tags"])
    return vectorizer, tfidf_matrix


def get_recommendations(df: pd.DataFrame, tfidf_matrix, title: str, top_n: int = 10):
    """
    Return the top_n movies most similar to `title` using cosine similarity
    on the TF-IDF matrix, computed lazily for just the query row (memory
    efficient — no need to materialize the full NxN similarity matrix).
    """
    matches = df.index[df["title"].str.lower() == title.lower()]
    if len(matches) == 0:
        return pd.DataFrame()
    idx = matches[0]

    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(sims))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

    rec_indices = [i for i, _ in sim_scores]
    rec_scores = [round(float(s) * 100, 1) for _, s in sim_scores]

    result = df.iloc[rec_indices].copy()
    result["match_score"] = rec_scores
    return result
