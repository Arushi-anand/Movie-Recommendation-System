from pathlib import Path
import json
import os
import pickle
import tomllib
import urllib.parse
import urllib.request

import pandas as pd
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from database import (
    authenticate_app_user,
    create_app_user,
    ensure_auth_schema,
    load_csv_recommender_data,
    load_recommender_data,
    load_user_ratings,
    save_movie_poster,
    save_recommendation_history,
    save_user_ratings,
)


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
RATING_OPTIONS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"


def read_database_url_from_file(path):
    if not path.exists():
        return None

    with path.open("rb") as secrets_file:
        secrets_data = tomllib.load(secrets_file)

    if secrets_data.get("database_url"):
        return secrets_data["database_url"]

    return secrets_data.get("connections", {}).get("mysql", {}).get("url")


def get_database_url():
    for variable_name in ("DATABASE_URL", "MYSQL_URL"):
        database_url = os.getenv(variable_name)
        if database_url:
            return database_url

    try:
        if st.secrets.get("database_url"):
            return st.secrets["database_url"]

        mysql_connection = st.secrets.get("connections", {}).get("mysql", {})
        if mysql_connection.get("url"):
            return mysql_connection["url"]
    except Exception:
        pass

    for secrets_path in (
        PROJECT_DIR / ".streamlit" / "secrets.toml",
        APP_DIR / ".streamlit" / "secrets.toml",
    ):
        database_url = read_database_url_from_file(secrets_path)
        if database_url:
            return database_url

    return None


def get_tmdb_api_key():
    for variable_name in ("TMDB_API_KEY", "tmdb_api_key"):
        api_key = os.getenv(variable_name)
        if api_key:
            return api_key

    try:
        if st.secrets.get("tmdb_api_key"):
            return st.secrets["tmdb_api_key"]
    except Exception:
        pass

    for secrets_path in (
        PROJECT_DIR / ".streamlit" / "secrets.toml",
        APP_DIR / ".streamlit" / "secrets.toml",
    ):
        if not secrets_path.exists():
            continue

        with secrets_path.open("rb") as secrets_file:
            secrets_data = tomllib.load(secrets_file)
        api_key = secrets_data.get("tmdb_api_key")
        if api_key:
            return api_key

    return None


def initialize_session_state():
    st.session_state.setdefault("auth_user", None)


def render_authentication(database_url):
    st.subheader("Sign in")
    st.caption("Create an account or sign in to save ratings and recommendations.")

    sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

    with sign_in_tab:
        with st.form("sign_in_form"):
            username = st.text_input("Username", key="sign_in_username")
            password = st.text_input("Password", type="password", key="sign_in_password")
            submitted = st.form_submit_button("Sign in", icon=":material/login:")

        if submitted:
            user = authenticate_app_user(database_url, username, password)
            if user:
                st.session_state.auth_user = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with sign_up_tab:
        with st.form("sign_up_form"):
            username = st.text_input("Username", key="sign_up_username")
            password = st.text_input("Password", type="password", key="sign_up_password")
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="sign_up_confirm_password",
            )
            submitted = st.form_submit_button("Create account", icon=":material/person_add:")

        if submitted:
            normalized_username = username.strip()
            if len(normalized_username) < 3:
                st.error("Username must be at least 3 characters.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                try:
                    st.session_state.auth_user = create_app_user(
                        database_url,
                        normalized_username,
                        password,
                    )
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))


def render_user_sidebar():
    user = st.session_state.auth_user
    with st.sidebar:
        st.caption("Signed in")
        st.write(user["username"])
        if st.button("Sign out", icon=":material/logout:"):
            st.session_state.auth_user = None
            st.rerun()


def render_rating_history(user_ratings):
    st.subheader("Your ratings")
    if user_ratings.empty:
        st.info("Your saved ratings will appear here after you get recommendations.")
        return

    display_ratings = user_ratings.rename(
        columns={
            "title": "Movie title",
            "genres": "Genres",
            "year": "Year",
            "rating": "Rating",
            "updated_at": "Last updated",
        }
    )
    st.dataframe(
        display_ratings[
            ["Movie title", "Rating", "Genres", "Year", "Last updated"]
        ],
        hide_index=True,
    )


@st.cache_data(ttl="7d", max_entries=1000, show_spinner=False)
def fetch_poster_image(poster_url):
    if not poster_url or not str(poster_url).startswith("https://image.tmdb.org/"):
        return None

    request = urllib.request.Request(
        poster_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            return response.read()
    except Exception:
        return None


@st.cache_data(ttl="7d", max_entries=1000, show_spinner=False)
def fetch_tmdb_poster_path(tmdb_id, api_key):
    if pd.isna(tmdb_id) or not api_key:
        return None

    query = urllib.parse.urlencode({"api_key": api_key, "language": "en-US"})
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    return payload.get("poster_path")


@st.cache_resource(show_spinner="Loading movie data...")
def load_movie_data(database_url):
    active_database_url = None
    data_source = "CSV files"

    if database_url:
        try:
            df_content, df_user = load_recommender_data(database_url)
            active_database_url = database_url
            data_source = "MySQL"
        except Exception as error:
            st.warning(
                "MySQL is configured, but the app could not load from it. "
                f"Using CSV files for now. Details: {error}"
            )
            df_content, df_user = load_csv_recommender_data(APP_DIR)
    else:
        df_content, df_user = load_csv_recommender_data(APP_DIR)

    similarity_matrix_path = APP_DIR / "movie_similarity_matrix.pkl"
    if not similarity_matrix_path.exists():
        st.error(
            "Missing movie similarity matrix. Run "
            "`python scripts/build_similarity_matrix.py` from the project root, "
            "then restart the app."
        )
        st.stop()

    with open(similarity_matrix_path, "rb") as similarity_file:
        content_similarity = pickle.load(similarity_file)

    if len(df_content) != content_similarity.shape[0]:
        if active_database_url:
            st.warning(
                "MySQL movie data does not match the similarity matrix yet. "
                "Using CSV files for now. Run scripts/init_mysql.py --replace "
                "after configuring MySQL."
            )
            df_content, df_user = load_csv_recommender_data(APP_DIR)
            active_database_url = None
            data_source = "CSV files"
        else:
            raise ValueError("Movie data does not match the similarity matrix.")

    df_content_sim = pd.DataFrame(
        content_similarity,
        index=df_content["title"].values,
        columns=df_content["title"].values,
    )
    return df_content, df_user, df_content_sim, data_source, active_database_url


def get_content_similar_movies(user_id, df_user, df_content, df_content_sim):
    df_current_user = df_user[df_user["user_id"] == user_id]
    user_watched_movies = df_current_user["title"].dropna().unique()

    if df_current_user.empty:
        return pd.DataFrame(columns=["title", "genres", "content_similarity"])

    user_mean_rating = df_current_user["rating"].mean()
    liked_movies = df_current_user.loc[
        df_current_user["rating"] >= user_mean_rating, "title"
    ].dropna().unique()

    similar_movie_scores = [
        df_content_sim[movie].drop(user_watched_movies, errors="ignore")
        for movie in liked_movies
        if movie in df_content_sim.columns
    ]

    if not similar_movie_scores:
        return pd.DataFrame(columns=["title", "genres", "content_similarity"])

    content_rec = (
        pd.concat(similar_movie_scores, axis=1)
        .sum(axis=1)
        .reset_index()
        .rename(columns={"index": "title", 0: "content_similarity"})
    )
    return (
        pd.merge(df_content[["title", "genres"]], content_rec, how="inner")
        .sort_values(by="content_similarity", ascending=False)
    )


def get_user_similar_movies(
    user_id,
    df_user,
    df_content,
    norm_user_item,
    df_user_sim,
    similarity_threshold,
):
    similar_users = (
        df_user_sim[df_user_sim[user_id] > similarity_threshold][user_id]
        .sort_values(ascending=False)[1:]
    )

    if similar_users.empty:
        return pd.DataFrame(columns=["title", "genres", "year", "user_similarity"])

    target_user_movies = norm_user_item.loc[[user_id]].dropna(axis=1, how="all")
    similar_user_movies = norm_user_item[
        norm_user_item.index.isin(similar_users.index)
    ].dropna(axis=1, how="all")
    similar_user_movies = similar_user_movies.drop(
        columns=target_user_movies.columns.intersection(similar_user_movies.columns)
    )

    movie_score = {}
    for movie in similar_user_movies.columns:
        movie_rating = similar_user_movies[movie]
        numerator = 0
        denominator = 0

        for similar_user_id in similar_users.index:
            if pd.notnull(movie_rating[similar_user_id]):
                weighted_score = similar_users[similar_user_id] * movie_rating[similar_user_id]
                numerator += weighted_score
                denominator += similar_users[similar_user_id]

        if denominator:
            movie_score[movie] = numerator / denominator

    if not movie_score:
        return pd.DataFrame(columns=["title", "genres", "year", "user_similarity"])

    movie_score = pd.DataFrame(movie_score.items(), columns=["title", "user_similarity"])
    user_rec = pd.merge(
        df_content[["title", "genres", "year"]],
        movie_score,
        how="inner",
    )
    return user_rec.sort_values(by=["user_similarity", "year"], ascending=False)


def hybrid_recommender(user_id, df_user, df_content, df_content_sim):
    user_item = df_user.pivot_table(values="rating", index="user_id", columns="title")
    norm_user_item = user_item.subtract(user_item.mean(axis=1), axis="rows")

    user_similarity = cosine_similarity(sparse.csr_matrix(norm_user_item.fillna(0)))
    df_user_sim = pd.DataFrame(user_similarity, index=user_item.index, columns=user_item.index)

    content_recs = get_content_similar_movies(user_id, df_user, df_content, df_content_sim)
    user_recs = get_user_similar_movies(
        user_id,
        df_user,
        df_content,
        norm_user_item,
        df_user_sim,
        similarity_threshold=0.1,
    )

    if content_recs.empty and user_recs.empty:
        return pd.DataFrame()

    if user_recs.empty:
        top_scores = content_recs.head(10).assign(
            similarity_score=content_recs["content_similarity"]
        )
    elif content_recs.empty:
        top_scores = user_recs.head(10).assign(
            similarity_score=user_recs["user_similarity"]
        )
    else:
        top_scores = pd.merge(content_recs, user_recs, on=["title", "genres"])
        top_scores["similarity_score"] = (
            top_scores["content_similarity"] + top_scores["user_similarity"]
        ) / 2
        top_scores = top_scores.sort_values(by="similarity_score", ascending=False).head(10)

    recommendations = pd.merge(
        df_content[
            ["movie_id", "title", "genres", "tmdb_id", "imdb_rating", "tmdb_rating", "poster_url"]
        ],
        top_scores[["title", "similarity_score"]],
        on="title",
    )
    recommendations = recommendations.rename(
        columns={
            "title": "Movie title",
            "movie_id": "Movie ID",
            "genres": "Genres",
            "tmdb_id": "TMDB ID",
            "imdb_rating": "IMDb rating",
            "tmdb_rating": "TMDB rating",
            "poster_url": "Poster URL",
            "similarity_score": "Similarity score",
        }
    )
    return recommendations.sort_values(by="Similarity score", ascending=False)


def build_rating_rows(user_id, selected_ratings, df_content):
    rows = []
    for movie, rating in selected_ratings:
        movie_row = df_content.loc[df_content["title"] == movie].iloc[0]
        rows.append(
            {
                "user_id": user_id,
                "rating": rating,
                "movie_id": movie_row["movie_id"],
                "title": movie,
                "genres": movie_row["genres"],
                "year": movie_row["year"],
            }
        )
    return pd.DataFrame(rows).drop_duplicates(subset=["movie_id"])


def merge_current_user_ratings(base_df_user, user_id, df_new_user):
    updated_movie_ids = df_new_user["movie_id"]
    current_user_rating_mask = (
        (base_df_user["user_id"] == user_id)
        & (base_df_user["movie_id"].isin(updated_movie_ids))
    )
    return pd.concat(
        [base_df_user[~current_user_rating_mask], df_new_user],
        ignore_index=True,
    )


def filter_recommendations(recommendations, excluded_titles):
    if recommendations.empty:
        return recommendations

    return recommendations[~recommendations["Movie title"].isin(excluded_titles)]


def ensure_recommendation_posters(recommendations, database_url):
    if recommendations.empty:
        return recommendations

    api_key = get_tmdb_api_key()
    if not api_key:
        return recommendations

    recommendations = recommendations.copy()
    updated_posters = False
    for index, movie in recommendations.iterrows():
        poster_url = movie.get("Poster URL")
        if pd.notna(poster_url) and poster_url:
            continue

        poster_path = fetch_tmdb_poster_path(movie.get("TMDB ID"), api_key)
        if not poster_path:
            continue

        poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        recommendations.at[index, "Poster URL"] = poster_url
        if database_url and pd.notna(movie.get("Movie ID")):
            save_movie_poster(
                database_url,
                int(movie["Movie ID"]),
                poster_path,
                poster_url,
            )
            updated_posters = True

    if updated_posters:
        load_movie_data.clear()

    return recommendations


def render_recommendations(recommendations):
    if recommendations.empty:
        st.info("No recommendations were found. Try rating a few more movies.")
    else:
        st.subheader("Recommendations")
        for row_start in range(0, len(recommendations), 5):
            columns = st.columns(5)
            for column, (_, movie) in zip(
                columns,
                recommendations.iloc[row_start : row_start + 5].iterrows(),
            ):
                with column:
                    poster_url = movie.get("Poster URL")
                    poster_image = fetch_poster_image(poster_url)
                    if poster_image:
                        st.image(poster_image)
                    else:
                        st.container(height=260, border=True).write("No poster")
                    st.markdown(f"**{movie['Movie title']}**")
                    st.caption(movie["Genres"])
                    st.caption(
                        f"IMDb {movie['IMDb rating']} | TMDB {movie['TMDB rating']}"
                    )
                    st.caption(f"Score {movie['Similarity score']:.3f}")


st.set_page_config(page_title="Movie recommendations", page_icon=":material/movie:")
initialize_session_state()
st.header("Personalized movie recommendations")

database_url = get_database_url()
if not database_url:
    st.error("Authentication requires MySQL. Configure `.streamlit/secrets.toml` first.")
    st.stop()

ensure_auth_schema(database_url)

if not st.session_state.auth_user:
    render_authentication(database_url)
    st.stop()

render_user_sidebar()

df_content, base_df_user, df_content_sim, data_source, active_database_url = load_movie_data(
    database_url
)
st.caption(f"Data source: {data_source}")
movie_options = df_content["title"].dropna().tolist()
user_id = st.session_state.auth_user["id"]
user_ratings = load_user_ratings(database_url, user_id)

recommendation_mode = st.segmented_control(
    "Recommendation mode",
    ["Saved history", "Current selection only"],
    default="Saved history",
)

if recommendation_mode == "Saved history":
    st.caption("Use all saved ratings in your account.")
    if len(user_ratings) < 3:
        st.info("Save at least 3 ratings to unlock history-based recommendations.")
    elif st.button("Get recommendations from my history", icon=":material/recommend:"):
        with st.spinner("Finding movies you might like..."):
            recommendations = hybrid_recommender(
                user_id,
                base_df_user,
                df_content,
                df_content_sim,
            )

        rated_titles = set(base_df_user.loc[base_df_user["user_id"] == user_id, "title"])
        recommendations = filter_recommendations(recommendations, rated_titles)
        recommendations = ensure_recommendation_posters(recommendations, active_database_url)

        if active_database_url:
            try:
                save_recommendation_history(active_database_url, user_id, recommendations)
            except Exception as error:
                st.warning(f"Recommendations worked, but saving history failed: {error}")

        render_recommendations(recommendations)
else:
    number = int(
        st.number_input(
            "How many movies would you like to use?",
            min_value=3,
            value=3,
            step=1,
        )
    )

    selected_user_data = []
    with st.form("ratings_form"):
        st.caption("Use this when you want recommendations like a specific set of movies.")
        for index in range(number):
            movie = st.selectbox(
                "Movie title",
                options=movie_options,
                key=f"movie_{index}",
            )
            rating = st.select_slider(
                "Rate the movie",
                options=RATING_OPTIONS,
                key=f"rating_{index}",
            )
            selected_user_data.append((movie, rating))

        save_selection = st.checkbox("Save these ratings to my history", value=True)
        submitted = st.form_submit_button("Get recommendations")

    if submitted:
        selected_movies = [movie for movie, _ in selected_user_data]
        if len(selected_movies) != len(set(selected_movies)):
            st.warning("Please choose each movie only once for better recommendations.")
            st.stop()

        df_new_user = build_rating_rows(user_id, selected_user_data, df_content)

        if save_selection:
            df_user = merge_current_user_ratings(base_df_user, user_id, df_new_user)
        else:
            df_user = pd.concat(
                [base_df_user[base_df_user["user_id"] != user_id], df_new_user],
                ignore_index=True,
            )

        with st.spinner("Finding movies you might like..."):
            recommendations = hybrid_recommender(
                user_id,
                df_user,
                df_content,
                df_content_sim,
            )

        recommendations = filter_recommendations(recommendations, set(selected_movies))
        recommendations = ensure_recommendation_posters(recommendations, active_database_url)

        if active_database_url:
            try:
                if save_selection:
                    save_user_ratings(active_database_url, user_id, df_new_user)
                    load_movie_data.clear()
                save_recommendation_history(active_database_url, user_id, recommendations)
            except Exception as error:
                st.warning(f"Recommendations worked, but saving to MySQL failed: {error}")

        render_recommendations(recommendations)

render_rating_history(load_user_ratings(database_url, user_id))
