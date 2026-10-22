"""
Movie Recommendation System — Data Preprocessing
==================================================
Standalone script that runs the full cleaning / feature-engineering
pipeline (see recommender.py) on movies.csv, saves the cleaned dataset,
and prints an EDA-style summary so you can sanity-check the results
before launching the Streamlit app.

Usage:
    python preprocess.py
"""

from recommender import load_and_clean, build_similarity_model

RAW_PATH = "movies.csv"
OUT_CSV = "movies_cleaned.csv"


def main():
    print("=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM — PREPROCESSING")
    print("=" * 60)

    print("\n[1/4] Loading raw dataset...")
    import pandas as pd
    raw = pd.read_csv(RAW_PATH)
    print(f"   Raw shape: {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"   Missing values per column (top 10):")
    print(raw.isnull().sum().sort_values(ascending=False).head(10).to_string())

    print("\n[2/4] Cleaning + feature engineering...")
    df = load_and_clean(RAW_PATH)
    print(f"   Cleaned shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Dropped {raw.shape[0] - df.shape[0]} duplicate/unusable rows")

    print("\n[3/4] Building TF-IDF content model...")
    vectorizer, tfidf_matrix = build_similarity_model(df)
    print(f"   Vocabulary size : {len(vectorizer.get_feature_names_out())}")
    print(f"   TF-IDF matrix   : {tfidf_matrix.shape[0]} movies x {tfidf_matrix.shape[1]} features")

    print("\n[4/4] Saving cleaned dataset...")
    df.to_csv(OUT_CSV, index=False)
    print(f"   Saved -> {OUT_CSV}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_genres = sorted(set(g for lst in df["genre_list"] for g in lst))
    print(f"Movies              : {len(df)}")
    print(f"Genres ({len(all_genres)})       : {', '.join(all_genres)}")
    print(f"Year range          : {df['release_year'].min()} - {df['release_year'].max()}")
    print(f"Avg rating          : {df['vote_average'].mean():.2f} / 10")
    print(f"Top 5 by weighted score:")
    top5 = df.sort_values("weighted_score", ascending=False).head(5)
    for _, row in top5.iterrows():
        print(f"   - {row['title']} ({row['release_year']}) — {row['weighted_score']:.2f}")
    print("\nPreprocessing complete. Run 'streamlit run app.py' to launch the UI.")


if __name__ == "__main__":
    main()
