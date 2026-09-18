from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "recommender_app"


CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id INT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    genres VARCHAR(255),
    year INT,
    tmdb_id INT NULL,
    imdb_id VARCHAR(32),
    tmdb_rating FLOAT,
    tmdb_votes INT,
    imdb_rating FLOAT,
    imdb_votes INT,
    body TEXT,
    sentiment_score FLOAT,
    weighted_rating FLOAT,
    poster_path VARCHAR(255),
    poster_url VARCHAR(500),
    INDEX idx_movies_title (title)
);

CREATE TABLE IF NOT EXISTS app_users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(80) NOT NULL UNIQUE,
    password_hash VARCHAR(255),
    role VARCHAR(32) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ratings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    rating DECIMAL(2, 1) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_ratings_user_movie (user_id, movie_id),
    INDEX idx_ratings_user_id (user_id),
    CONSTRAINT fk_ratings_movie
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS recommendation_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    movie_id INT NOT NULL,
    score FLOAT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_recommendation_history_user_id (user_id),
    CONSTRAINT fk_recommendation_history_movie
        FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
        ON DELETE CASCADE
);
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create MySQL tables and import the recommender CSV data."
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL"),
        help=(
            "SQLAlchemy MySQL URL, for example "
            "mysql+pymysql://user:password@localhost:3306/movie_recommender"
        ),
    )
    parser.add_argument(
        "--content-csv",
        type=Path,
        default=APP_DIR / "clean_content.csv",
        help="Path to clean_content.csv.",
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=APP_DIR / "ratings_title.csv",
        help="Path to ratings_title.csv.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Clear existing movies/ratings/recommendation history before importing.",
    )
    return parser.parse_args()


def create_schema(engine):
    with engine.begin() as connection:
        for statement in CREATE_TABLES_SQL.strip().split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(text(statement))
        for statement in (
            "ALTER TABLE movies ADD COLUMN poster_path VARCHAR(255)",
            "ALTER TABLE movies ADD COLUMN poster_url VARCHAR(500)",
        ):
            try:
                connection.execute(text(statement))
            except Exception:
                pass


def import_movies(engine, content_csv: Path, replace: bool):
    df_content = pd.read_csv(content_csv)
    movie_columns = [
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
    ]
    df_content = df_content[movie_columns].drop_duplicates(subset=["movie_id"])

    if replace:
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM recommendation_history"))
            connection.execute(text("DELETE FROM ratings"))
            connection.execute(text("DELETE FROM movies"))

    df_content.to_sql("movies", engine, if_exists="append", index=False, chunksize=1000)
    return len(df_content)


def import_ratings(engine, ratings_csv: Path):
    df_ratings = pd.read_csv(ratings_csv)
    df_ratings = df_ratings.rename(columns={"userId": "user_id", "movieId": "movie_id"})
    df_ratings = df_ratings[["user_id", "movie_id", "rating"]].drop_duplicates(
        subset=["user_id", "movie_id"]
    )

    movie_ids = pd.read_sql("SELECT movie_id FROM movies", engine)["movie_id"]
    valid_movie_ids = set(movie_ids)
    original_count = len(df_ratings)
    df_ratings = df_ratings[df_ratings["movie_id"].isin(valid_movie_ids)]
    skipped_count = original_count - len(df_ratings)

    df_ratings.to_sql("ratings", engine, if_exists="append", index=False, chunksize=1000)
    return len(df_ratings), skipped_count


def set_app_user_auto_increment(engine):
    with engine.begin() as connection:
        max_rating_user_id = connection.execute(
            text("SELECT COALESCE(MAX(user_id), 0) FROM ratings")
        ).scalar_one()
        max_app_user_id = connection.execute(
            text("SELECT COALESCE(MAX(id), 0) FROM app_users")
        ).scalar_one()
        next_user_id = max(10_000, max_rating_user_id + 1, max_app_user_id + 1)
        connection.execute(text(f"ALTER TABLE app_users AUTO_INCREMENT = {int(next_user_id)}"))
    return next_user_id


def main():
    args = parse_args()
    if not args.database_url:
        raise SystemExit(
            "Missing database URL. Set DATABASE_URL or pass --database-url."
        )

    engine = create_engine(args.database_url, pool_pre_ping=True, future=True)
    create_schema(engine)
    movie_count = import_movies(engine, args.content_csv, args.replace)
    rating_count, skipped_rating_count = import_ratings(engine, args.ratings_csv)
    next_user_id = set_app_user_auto_increment(engine)

    print(f"Imported {movie_count:,} movies.")
    print(f"Imported {rating_count:,} ratings.")
    if skipped_rating_count:
        print(f"Skipped {skipped_rating_count:,} ratings for movies not in clean_content.csv.")
    print(f"Next app user id starts at {next_user_id:,}.")
    print("MySQL setup complete.")


if __name__ == "__main__":
    main()
