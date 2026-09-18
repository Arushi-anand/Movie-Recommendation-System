from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "recommender_app"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the movie content similarity matrix used by the Streamlit app."
    )
    parser.add_argument(
        "--content-csv",
        type=Path,
        default=APP_DIR / "clean_content.csv",
        help="Path to clean_content.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=APP_DIR / "movie_similarity_matrix.pkl",
        help="Where to save the generated pickle file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df_content = pd.read_csv(args.content_csv)

    if "body" not in df_content.columns:
        raise SystemExit("clean_content.csv must contain a 'body' column.")

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df_content["body"].fillna(""))
    similarity_matrix = cosine_similarity(tfidf_matrix)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(similarity_matrix, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Built similarity matrix: {args.output}")
    print(f"Shape: {similarity_matrix.shape[0]:,} x {similarity_matrix.shape[1]:,}")


if __name__ == "__main__":
    main()
