from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from sqlalchemy import create_engine, text


ROOT_DIR = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT_DIR / ".streamlit" / "secrets.toml"
TMDB_API_URL = "https://api.themoviedb.org/3/movie/{tmdb_id}"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"


def load_project_secrets():
    if not SECRETS_PATH.exists():
        return {}

    with SECRETS_PATH.open("rb") as secrets_file:
        return tomllib.load(secrets_file)


def get_secret(name: str):
    secrets = load_project_secrets()
    return os.getenv(name.upper()) or os.getenv(name) or secrets.get(name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch TMDB poster URLs for movies missing posters."
    )
    parser.add_argument(
        "--database-url",
        default=get_secret("database_url") or os.getenv("MYSQL_URL"),
        help="SQLAlchemy MySQL URL. Defaults to DATABASE_URL or .streamlit/secrets.toml.",
    )
    parser.add_argument(
        "--tmdb-api-key",
        default=get_secret("tmdb_api_key"),
        help="TMDB API key. Defaults to TMDB_API_KEY or .streamlit/secrets.toml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=250,
        help="Maximum movies to fetch in this run.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between TMDB requests.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=1,
        help="Retry attempts for transient network errors.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8,
        help="Seconds to wait for each TMDB request before retrying.",
    )
    return parser.parse_args()


def ensure_poster_columns(engine):
    with engine.begin() as connection:
        for statement in (
            "ALTER TABLE movies ADD COLUMN poster_path VARCHAR(255)",
            "ALTER TABLE movies ADD COLUMN poster_url VARCHAR(500)",
        ):
            try:
                connection.execute(text(statement))
            except Exception:
                pass


def load_movies_missing_posters(engine, limit: int):
    with engine.connect() as connection:
        return (
            connection.execute(
                text(
                    """
                    SELECT movie_id, title, tmdb_id
                    FROM movies
                    WHERE tmdb_id IS NOT NULL
                      AND (poster_url IS NULL OR poster_url = '')
                    ORDER BY movie_id
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            )
            .mappings()
            .all()
        )


def fetch_poster_path(tmdb_id: int, api_key: str, timeout: float):
    query = urllib.parse.urlencode({"api_key": api_key, "language": "en-US"})
    url = f"{TMDB_API_URL.format(tmdb_id=int(tmdb_id))}?{query}"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise

    return payload.get("poster_path")


def fetch_poster_path_with_retries(
    tmdb_id: int,
    api_key: str,
    retries: int,
    timeout: float,
):
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fetch_poster_path(tmdb_id, api_key, timeout)
        except urllib.error.HTTPError as error:
            if error.code in {401, 403, 404}:
                raise
            last_error = error
        except urllib.error.URLError as error:
            last_error = error
        except ConnectionResetError as error:
            last_error = error

        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    raise last_error


def save_poster(engine, movie_id: int, poster_path: str | None):
    poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
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


def main():
    args = parse_args()
    if not args.database_url:
        raise SystemExit("Missing database URL. Configure .streamlit/secrets.toml first.")
    if not args.tmdb_api_key:
        raise SystemExit(
            "Missing TMDB API key. Add tmdb_api_key to .streamlit/secrets.toml."
        )

    engine = create_engine(args.database_url, pool_pre_ping=True, future=True)
    ensure_poster_columns(engine)
    movies = load_movies_missing_posters(engine, args.limit)

    fetched = 0
    missing = 0
    failed = 0
    for movie in movies:
        try:
            poster_path = fetch_poster_path_with_retries(
                movie["tmdb_id"],
                args.tmdb_api_key,
                args.retries,
                args.timeout,
            )
            save_poster(engine, movie["movie_id"], poster_path)
            if poster_path:
                fetched += 1
            else:
                missing += 1
        except Exception as error:
            failed += 1
            print(f"Failed: {movie['title']} ({movie['tmdb_id']}): {error}")

        time.sleep(args.sleep)

    print(f"Processed {len(movies):,} movies.")
    print(f"Fetched {fetched:,} posters.")
    print(f"Missing posters for {missing:,} movies.")
    print(f"Failed {failed:,} movies.")


if __name__ == "__main__":
    main()
