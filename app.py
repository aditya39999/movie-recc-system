"""
CineMatch — Content-Based Movie Recommendation System
========================================================
A Streamlit interface for a TF-IDF + cosine-similarity content-based
recommender, built on top of the TMDB 5000 movies dataset.

Run:
    streamlit run app.py
"""

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from recommender import load_and_clean, build_similarity_model, get_recommendations

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

RAW_PATH = "movies.csv"
PLACEHOLDER_POSTER = "https://placehold.co/300x450/1a1a2e/e94560?text=No+Poster"

# ----------------------------------------------------------------------------
# Custom CSS — cinematic dark theme
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&family=Bebas+Neue&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    .stApp {
        background: radial-gradient(circle at top left, #1b1035 0%, #0d0b1e 45%, #0a0a14 100%);
        color: #f1f1f6;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14102b 0%, #0c0a1c 100%);
        border-right: 1px solid rgba(233, 69, 96, 0.15);
    }

    .hero {
        padding: 2.2rem 2rem 1.8rem 2rem;
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(233,69,96,0.18), rgba(83,52,255,0.18));
        border: 1px solid rgba(233,69,96,0.25);
        margin-bottom: 1.6rem;
        text-align: center;
    }
    .hero h1 {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.4rem;
        letter-spacing: 3px;
        background: linear-gradient(90deg, #ff6b81, #ffb86c, #f0a6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero p { color: #c9c9d9; font-size: 1.05rem; margin-top: 0.3rem; }

    .movie-card {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.9rem;
        transition: transform 0.18s ease, border-color 0.18s ease;
        height: 100%;
    }
    .movie-card:hover {
        transform: translateY(-6px);
        border-color: rgba(233,69,96,0.55);
    }
    .movie-title {
        font-weight: 700;
        font-size: 1.02rem;
        margin: 0.55rem 0 0.15rem 0;
        color: #fff;
        line-height: 1.25;
        min-height: 2.5em;
    }
    .movie-meta { color: #9c9cb5; font-size: 0.82rem; margin-bottom: 0.4rem; }
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 2px 3px 2px 0;
    }
    .badge-genre { background: rgba(120, 90, 255, 0.22); color: #c9b8ff; border: 1px solid rgba(120,90,255,0.4);}
    .badge-rating { background: rgba(255, 184, 108, 0.18); color: #ffcf94; border: 1px solid rgba(255,184,108,0.4);}
    .badge-match { background: rgba(233, 69, 96, 0.22); color: #ff9baa; border: 1px solid rgba(233,69,96,0.45);}

    .section-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.9rem;
        letter-spacing: 2px;
        color: #ffd8a8;
        border-left: 5px solid #e94560;
        padding-left: 12px;
        margin: 1.4rem 0 0.8rem 0;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 10px;
    }

    .stButton>button {
        background: linear-gradient(90deg, #e94560, #ff6b81);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: filter 0.15s ease;
    }
    .stButton>button:hover { filter: brightness(1.15); }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-thumb { background: #3a2f5e; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data loading (cached — only runs once per server session)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_clean_data():
    return load_and_clean(RAW_PATH)


@st.cache_resource(show_spinner=False)
def get_model(df_hash):
    df = get_clean_data()
    vectorizer, tfidf_matrix = build_similarity_model(df)
    return vectorizer, tfidf_matrix


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster_by_id(tmdb_id, api_key: str):
    """Fetch a poster straight from TMDB using the movie's real TMDB id.
    This is a direct lookup (not a fuzzy title search), so it works for
    almost every movie in the dataset."""
    if not api_key or pd.isna(tmdb_id):
        return None
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=4,
        )
        data = r.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poster_by_title(title: str, api_key: str, year=None):
    """Fallback: fuzzy title search, only used if the ID lookup fails
    (e.g. a movie was removed/merged on TMDB since the dataset was built)."""
    if not api_key:
        return None
    try:
        params = {"api_key": api_key, "query": title}
        if year:
            params["year"] = int(year)
        r = requests.get("https://api.themoviedb.org/3/search/movie", params=params, timeout=4)
        data = r.json()
        results = data.get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w342{results[0]['poster_path']}"
    except Exception:
        return None
    return None


with st.spinner("Loading & preprocessing dataset..."):
    df = get_clean_data()
    vectorizer, tfidf_matrix = get_model(len(df))

all_genres = sorted({g for lst in df["genre_list"] for g in lst})

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Content-based recommender · TF-IDF + Cosine Similarity")
    st.divider()

    st.markdown("### 🔑 Poster images (optional)")
    tmdb_key = st.text_input(
        "TMDB API key", type="password",
        help="Paste a free TMDB API key to show real posters. Without it, "
             "placeholder art is shown instead. Get one at themoviedb.org.",
    )
    st.divider()

    st.markdown("### ⚙️ Recommendation settings")
    top_n = st.slider("Number of recommendations", 4, 20, 10)
    st.divider()

    st.markdown("### 📊 Dataset")
    st.metric("Total movies", f"{len(df):,}")
    st.metric("Genres", len(all_genres))
    st.caption(f"Years {df['release_year'].min()}–{df['release_year'].max()}")


def poster_url(title, year, tmdb_id=None):
    if not tmdb_key:
        return PLACEHOLDER_POSTER
    url = fetch_poster_by_id(tmdb_id, tmdb_key)
    if not url:
        url = fetch_poster_by_title(title, tmdb_key, year)
    return url or PLACEHOLDER_POSTER


def render_card(row, show_match=False, col=None):
    target = col if col is not None else st
    with target:
        img = poster_url(row["title"], row.get("release_year"), row.get("id"))
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown(f'<div class="movie-title">{row["title"]}</div>', unsafe_allow_html=True)
        year = row.get("release_year", "")
        director = row.get("director", "") or "Unknown"
        st.markdown(f'<div class="movie-meta">{year} · dir. {director}</div>', unsafe_allow_html=True)

        badges = f'<span class="badge badge-rating">⭐ {row["vote_average"]:.1f}</span>'
        if show_match and "match_score" in row:
            badges += f'<span class="badge badge-match">🎯 {row["match_score"]}% match</span>'
        st.markdown(badges, unsafe_allow_html=True)

        genres = row["genre_list"][:3] if isinstance(row["genre_list"], list) else []
        genre_html = "".join(f'<span class="badge badge-genre">{g}</span>' for g in genres)
        st.markdown(genre_html, unsafe_allow_html=True)

        overview = row.get("overview", "") or "No synopsis available."
        short_overview = overview if len(overview) <= 140 else overview[:137] + "..."
        st.caption(short_overview)
        st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Hero header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 CINEMATCH</h1>
    <p>Discover your next favorite film — powered by content-based AI recommendations</p>
</div>
""", unsafe_allow_html=True)

tab_recommend, tab_explore, tab_top, tab_insights = st.tabs(
    ["🎯 Get Recommendations", "🔍 Explore & Filter", "🏆 Top Rated", "📈 Insights"]
)

# ----------------------------------------------------------------------------
# TAB 1 — Recommend
# ----------------------------------------------------------------------------
with tab_recommend:
    st.markdown('<div class="section-title">Pick a movie you love</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        movie_choice = st.selectbox(
            "Search for a movie", options=df["title"].sort_values().tolist(),
            index=None, placeholder="Type to search (e.g. 'The Dark Knight')...",
            label_visibility="collapsed",
        )
    with c2:
        go_btn = st.button("✨ Recommend", use_container_width=True)

    if movie_choice:
        src = df[df["title"] == movie_choice].iloc[0]
        with st.container():
            cols = st.columns([1, 4])
            with cols[0]:
                st.image(poster_url(src["title"], src["release_year"], src.get("id")), use_container_width=True)
            with cols[1]:
                st.markdown(f"#### {src['title']} ({src['release_year']})")
                st.write(src["overview"] or "_No synopsis available._")
                st.markdown(
                    f'<span class="badge badge-rating">⭐ {src["vote_average"]:.1f}/10</span>'
                    + "".join(f'<span class="badge badge-genre">{g}</span>' for g in src["genre_list"]),
                    unsafe_allow_html=True,
                )
        st.divider()

        with st.spinner("Finding movies with similar DNA..."):
            recs = get_recommendations(df, tfidf_matrix, movie_choice, top_n=top_n)

        if recs.empty:
            st.warning("No recommendations found for this title.")
        else:
            st.markdown(f'<div class="section-title">Because you liked "{movie_choice}"</div>', unsafe_allow_html=True)
            n_cols = 5
            rows = [recs.iloc[i:i + n_cols] for i in range(0, len(recs), n_cols)]
            for chunk in rows:
                cols = st.columns(n_cols)
                for i, (_, r) in enumerate(chunk.iterrows()):
                    render_card(r, show_match=True, col=cols[i])
    else:
        st.info("👆 Select a movie above to get personalized recommendations based on genre, cast, director, keywords and plot similarity.")

# ----------------------------------------------------------------------------
# TAB 2 — Explore & Filter
# ----------------------------------------------------------------------------
with tab_explore:
    st.markdown('<div class="section-title">Browse the catalog</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        genre_filter = st.multiselect("Genre", all_genres)
    with f2:
        year_range = st.slider(
            "Release year", int(df["release_year"].min()), int(df["release_year"].max()),
            (2000, int(df["release_year"].max())),
        )
    with f3:
        min_rating = st.slider("Minimum rating", 0.0, 10.0, 6.0, 0.5)
    with f4:
        sort_by = st.selectbox("Sort by", ["Popularity", "Rating", "Release Year (Newest)", "Title (A-Z)"])

    filtered = df[
        (df["release_year"].between(*year_range)) &
        (df["vote_average"] >= min_rating)
    ].copy()
    if genre_filter:
        filtered = filtered[filtered["genre_list"].apply(lambda lst: any(g in lst for g in genre_filter))]

    sort_map = {
        "Popularity": ("popularity", False),
        "Rating": ("vote_average", False),
        "Release Year (Newest)": ("release_year", False),
        "Title (A-Z)": ("title", True),
    }
    sort_col, ascending = sort_map[sort_by]
    filtered = filtered.sort_values(sort_col, ascending=ascending)

    st.caption(f"Showing {min(len(filtered), 25)} of {len(filtered)} matching movies")

    if filtered.empty:
        st.warning("No movies match these filters — try widening your criteria.")
    else:
        page = filtered.head(25)
        n_cols = 5
        rows = [page.iloc[i:i + n_cols] for i in range(0, len(page), n_cols)]
        for chunk in rows:
            cols = st.columns(n_cols)
            for i, (_, r) in enumerate(chunk.iterrows()):
                render_card(r, col=cols[i])

# ----------------------------------------------------------------------------
# TAB 3 — Top Rated
# ----------------------------------------------------------------------------
with tab_top:
    st.markdown('<div class="section-title">Critically acclaimed (Bayesian weighted score)</div>', unsafe_allow_html=True)
    st.caption("Ranked using an IMDB-style weighted formula that balances rating with vote count, "
               "so a 9.0 from 50k votes beats a 10.0 from 3 votes.")
    top_movies = df.sort_values("weighted_score", ascending=False).head(20)
    n_cols = 5
    rows = [top_movies.iloc[i:i + n_cols] for i in range(0, len(top_movies), n_cols)]
    for chunk in rows:
        cols = st.columns(n_cols)
        for i, (_, r) in enumerate(chunk.iterrows()):
            render_card(r, col=cols[i])

# ----------------------------------------------------------------------------
# TAB 4 — Insights (EDA)
# ----------------------------------------------------------------------------
with tab_insights:
    st.markdown('<div class="section-title">Dataset insights</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Movies", f"{len(df):,}")
    m2.metric("Avg. rating", f"{df['vote_average'].mean():.2f} / 10")
    m3.metric("Avg. runtime", f"{df['runtime'].mean():.0f} min")
    m4.metric("Total revenue", f"${df['revenue'].sum()/1e9:.1f}B")

    chart_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f1f6", margin=dict(t=40, l=10, r=10, b=10),
    )

    cA, cB = st.columns(2)
    with cA:
        genre_counts = pd.Series([g for lst in df["genre_list"] for g in lst]).value_counts().head(12)
        fig = px.bar(
            x=genre_counts.values, y=genre_counts.index, orientation="h",
            title="Most common genres", color=genre_counts.values,
            color_continuous_scale=["#5334ff", "#e94560"],
            labels={"x": "Number of movies", "y": ""},
        )
        fig.update_layout(**chart_theme, showlegend=False, coloraxis_showscale=False)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    with cB:
        fig = px.histogram(
            df, x="vote_average", nbins=30, title="Rating distribution",
            color_discrete_sequence=["#ff6b81"],
        )
        fig.update_layout(**chart_theme)
        st.plotly_chart(fig, use_container_width=True)

    cC, cD = st.columns(2)
    with cC:
        by_year = df.groupby("release_year").size().reset_index(name="count")
        by_year = by_year[by_year["release_year"] >= 1960]
        fig = px.area(
            by_year, x="release_year", y="count", title="Movies released per year",
            color_discrete_sequence=["#f0a6ff"],
        )
        fig.update_layout(**chart_theme)
        st.plotly_chart(fig, use_container_width=True)

    with cD:
        fig = px.scatter(
            df.sample(min(800, len(df)), random_state=1), x="budget", y="revenue",
            hover_name="title", title="Budget vs. Revenue",
            color="vote_average", color_continuous_scale=["#5334ff", "#ff6b81", "#ffd8a8"],
            labels={"budget": "Budget ($)", "revenue": "Revenue ($)"},
        )
        fig.update_layout(**chart_theme)
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with a TF-IDF + cosine-similarity content-based model on genres, keywords, "
           "overview, cast and director. No external ratings data or collaborative filtering used.")
