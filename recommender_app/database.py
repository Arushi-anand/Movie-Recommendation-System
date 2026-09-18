from __future__ import annotations

import hashlib
import hmac
import secrets
from pathlib import Path

import pandas as pd


try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import IntegrityError
except ImportError:  # pragma: no cover - exercised only when db extras are missing.
    create_engine = None
    text = None
    IntegrityError = Exception


CONTENT_COLUMNS = [
    "movie_id",
    "title",
    "genres",
    "year",
    "tmdb_id",
    "imdb_id",
    "tmdb_rating",
    "tmdb_votes",
    "imdb_rating",
    "imdb_votes",
    "body",
    "sentiment_score",
    "weighted_rating",
    "poster_path",
    "poster_url",
]

PASSWORD_ITERATIONS = 260_000
PASSWORD_SCHEME = "pbkdf2_sha256"
APP_USER_ID_START = 10_000


def _require_sqlalchemy():
    if create_engine is None:
        raise RuntimeError(
            "Database support requires SQLAlchemy and PyMySQL. "
            "Install them with: pip install -r requirements.txt"
        )


def get_engine(database_url: str):
    _require_sqlalchemy()
    return create_engine(database_url, pool_pre_ping=True, future=True)


def normalize_username(username: str) -> str:
    return username.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"{PASSWORD_SCHEME}${PASSWORD_ITERATIONS}${salt}${password_hash}"


def verify_password(password: str, stored_password_hash: str) -> bool:
    try:
        scheme, iterations, salt, expected_hash = stored_password_hash.split("$", 3)
    except ValueError:
        return False

    if scheme != PASSWORD_SCHEME:
        return False

    actual_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        int(iterations),
    ).hex()
    return hmac.compare_digest(actual_hash, expected_hash)


def ensure_auth_schema(database_url: str) -> None:
    engine = get_engine(database_url)
    with engine.begin() as connection:
        for statement in (
            "ALTER TABLE movies ADD COLUMN poster_path VARCHAR(255)",
            "ALTER TABLE movies ADD COLUMN poster_url VARCHAR(500)",
        ):
            try:
                connection.execute(text(statement))
            except Exception:
                pass

        max_rating_user_id = connection.execute(
            text("SELECT COALESCE(MAX(user_id), 0) FROM ratings")
        ).scalar_one()
        max_app_user_id = connection.execute(
            text("SELECT COALESCE(MAX(id), 0) FROM app_users")
        ).scalar_one()
        next_user_id = max(APP_USER_ID_START, max_rating_user_id + 1, max_app_user_id + 1)
        connection.execute(text(f"ALTER TABLE app_users AUTO_INCREMENT = {int(next_user_id)}"))


def create_app_user(database_url: str, username: str, password: str) -> dict:
    normalized_username = normalize_username(username)
    engine = get_engine(database_url)

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO app_users (username, password_hash, role)
                    VALUES (:username, :password_hash, 'user')
                    """
                ),
                {
                    "username": normalized_username,
                    "password_hash": hash_password(password),
                },
            )
            user_id = connection.execute(text("SELECT LAST_INSERT_ID()")).scalar_one()
    except IntegrityError as error:
        raise ValueError("That username is already taken.") from error

    return {"id": int(user_id), "username": normalized_username, "role": "user"}


def authenticate_app_user(database_url: str, username: str, password: str) -> dict | None:
    normalized_username = normalize_username(username)
    engine = get_engine(database_url)

    with engine.connect() as connection:
        user = (
            connection.execute(
                text(
                    """
                    SELECT id, username, password_hash, role
                    FROM app_users
                    WHERE username = :username
                    """
                ),
                {"username": normalized_username},
            )
            .mappings()
            .first()
        )

    if not user or not user["password_hash"]:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    return {
        "id": int(user["id"]),
        "username": user["username"],
        "role": user["role"],
    }


def load_recommender_data(database_url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    engine = get_engine(database_url)

    df_content = pd.read_sql(
        """
        SELECT movie_id, title, genres, year, tmdb_id, imdb_id,
               tmdb_rating, tmdb_votes, imdb_rating, imdb_votes,
               body, sentiment_score, weighted_rating, poster_path, poster_url
        FROM movies
        ORDER BY movie_id
        """,
        engine,
    )

    df_user = pd.read_sql(
        """
        SELECT r.user_id, r.movie_id, r.rating, m.title, m.genres, m.year
        FROM ratings r
        JOIN movies m ON m.movie_id = r.movie_id
        ORDER BY r.user_id, r.movie_id
        """,
        engine,
    )

    return df_content, df_user


def load_user_ratings(database_url: str, user_id: int) -> pd.DataFrame:
    engine = get_engine(database_url)
    return pd.read_sql(
        text(
            """
        SELECT
            r.movie_id,
            m.title,
            m.genres,
            m.year,
            r.rating,
            r.updated_at
        FROM ratings r
        JOIN movies m ON m.movie_id = r.movie_id
        WHERE r.user_id = :user_id
        ORDER BY r.updated_at DESC, m.title
        """
        ),
        engine,
        params={"user_id": int(user_id)},
    )


def save_user_ratings(database_url: str, user_id: int, ratings: pd.DataFrame) -> None:
    if ratings.empty:
        return

    engine = get_engine(database_url)
    statement = text(
        """
        INSERT INTO ratings (user_id, movie_id, rating)
        VALUES (:user_id, :movie_id, :rating)
        ON DUPLICATE KEY UPDATE rating = VALUES(rating)
        """
    )

    rows = [
        {
            "user_id": int(user_id),
            "movie_id": int(row.movie_id),
            "rating": float(row.rating),
        }
        for row in ratings.itertuples(index=False)
    ]

    with engine.begin() as connection:
        connection.execute(statement, rows)


def save_recommendation_history(
    database_url: str,
    user_id: int,
    recommendations: pd.DataFrame,
) -> None:
    if recommendations.empty:
        return

    engine = get_engine(database_url)
    with engine.begin() as connection:
        rows = []
        for _, row in recommendations.iterrows():
            movie_id = row.get("Movie ID")
            if pd.isna(movie_id):
                continue
            rows.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    "score": float(row["Similarity score"]),
                }
            )

        if rows:
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation_history (user_id, movie_id, score)
                    VALUES (:user_id, :movie_id, :score)
                    """
                ),
                rows,
            )


def save_movie_poster(
    database_url: str,
    movie_id: int,
    poster_path: str | None,
    poster_url: str | None,
) -> None:
    engine = get_engine(database_url)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                UPDATE movies
                SET poster_path = :poster_path,
                    poster_url = :poster_url
                WHERE movie_id = :movie_id
                """
            ),
            {
                "movie_id": int(movie_id),
                "poster_path": poster_path,
                "poster_url": poster_url,
            },
        )


def load_csv_recommender_data(app_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_content = pd.read_csv(app_dir / "clean_content.csv")
    df_user = pd.read_csv(app_dir / "ratings_title.csv")
    df_user = df_user.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    return df_content, df_user
