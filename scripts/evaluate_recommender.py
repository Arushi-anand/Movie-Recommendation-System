from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "recommender_app"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the hybrid recommender with precision@K and recall@K."
    )
    parser.add_argument("--k", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--liked-threshold",
        type=float,
        default=4.0,
        help="Ratings at or above this value are treated as relevant movies.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of each user's liked movies to hide for testing.",
    )
    parser.add_argument(
        "--min-train-ratings",
        type=int,
        default=5,
        help="Minimum ratings a user must keep in training after holdout.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=100,
        help="Maximum users to evaluate. Use 0 for all eligible users.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for selecting held-out movies.",
    )
    return parser.parse_args()


def load_data():
    content_path = APP_DIR / "clean_content.csv"
    ratings_path = APP_DIR / "ratings_title.csv"
    similarity_path = APP_DIR / "movie_similarity_matrix.pkl"

    if not similarity_path.exists():
        raise SystemExit(
            "Missing recommender_app/movie_similarity_matrix.pkl. "
            "Run: python scripts/build_similarity_matrix.py"
        )

    df_content = pd.read_csv(content_path)
    df_user = pd.read_csv(ratings_path).rename(
        columns={"userId": "user_id", "movieId": "movie_id"}
    )

    with similarity_path.open("rb") as similarity_file:
        content_similarity = pickle.load(similarity_file)

    return df_content, df_user, content_similarity


def content_recommendations(user_id, df_user, df_content, content_similarity, title_to_index):
    user_ratings = df_user[df_user["user_id"] == user_id]
    watched_titles = set(user_ratings["title"].dropna())
    if user_ratings.empty:
        return pd.DataFrame(columns=["title", "genres", "content_similarity"])

    user_mean_rating = user_ratings["rating"].mean()
    liked_titles = user_ratings.loc[
        user_ratings["rating"] >= user_mean_rating, "title"
    ].dropna()
    liked_indices = [title_to_index[title] for title in liked_titles if title in title_to_index]
    if not liked_indices:
        return pd.DataFrame(columns=["title", "genres", "content_similarity"])

    scores = np.asarray(content_similarity[:, liked_indices]).sum(axis=1)
    if sparse.issparse(scores):
        scores = np.asarray(scores).ravel()
    else:
        scores = scores.ravel()

    recommendations = df_content[["title", "genres"]].copy()
    recommendations["content_similarity"] = scores
    recommendations = recommendations[~recommendations["title"].isin(watched_titles)]
    return recommendations.sort_values("content_similarity", ascending=False)


def collaborative_recommendations(user_id, df_user, df_content, similarity_threshold=0.1):
    user_item = df_user.pivot_table(values="rating", index="user_id", columns="title")
    if user_id not in user_item.index:
        return pd.DataFrame(columns=["title", "genres", "year", "user_similarity"])

    norm_user_item = user_item.subtract(user_item.mean(axis=1), axis="rows")
    target_vector = norm_user_item.loc[[user_id]].fillna(0)
    user_vectors = norm_user_item.fillna(0)

    similarities = cosine_similarity(target_vector, user_vectors).ravel()
    similar_users = pd.Series(similarities, index=user_vectors.index)
    similar_users = similar_users[similar_users > similarity_threshold].drop(
        labels=[user_id], errors="ignore"
    )
    similar_users = similar_users.sort_values(ascending=False)
    if similar_users.empty:
        return pd.DataFrame(columns=["title", "genres", "year", "user_similarity"])

    target_rated_titles = user_item.loc[user_id].dropna().index
    candidate_ratings = norm_user_item.loc[similar_users.index]
    candidate_ratings = candidate_ratings.drop(
        columns=target_rated_titles.intersection(candidate_ratings.columns)
    )
    candidate_ratings = candidate_ratings.dropna(axis=1, how="all")
    if candidate_ratings.empty:
        return pd.DataFrame(columns=["title", "genres", "year", "user_similarity"])

    weighted_sum = candidate_ratings.mul(similar_users, axis=0).sum(axis=0, skipna=True)
    weight_sum = candidate_ratings.notna().mul(similar_users, axis=0).sum(axis=0)
    movie_scores = (weighted_sum / weight_sum).dropna()

    recommendations = pd.DataFrame(
        {"title": movie_scores.index, "user_similarity": movie_scores.values}
    )
    recommendations = pd.merge(
        df_content[["title", "genres", "year"]],
        recommendations,
        on="title",
        how="inner",
    )
    return recommendations.sort_values(["user_similarity", "year"], ascending=False)


def hybrid_recommendations(user_id, df_user, df_content, content_similarity, title_to_index, k):
    content_recs = content_recommendations(
        user_id, df_user, df_content, content_similarity, title_to_index
    )
    user_recs = collaborative_recommendations(user_id, df_user, df_content)

    if content_recs.empty and user_recs.empty:
        return []
    if user_recs.empty:
        top_scores = content_recs.assign(
            similarity_score=content_recs["content_similarity"]
        )
    elif content_recs.empty:
        top_scores = user_recs.assign(similarity_score=user_recs["user_similarity"])
    else:
        top_scores = pd.merge(content_recs, user_recs, on=["title", "genres"])
        top_scores["similarity_score"] = (
            top_scores["content_similarity"] + top_scores["user_similarity"]
        ) / 2

    return (
        top_scores.sort_values("similarity_score", ascending=False)
        .head(k)["title"]
        .tolist()
    )


def evaluate_user(
    user_id,
    df_user,
    df_content,
    content_similarity,
    title_to_index,
    args,
    rng,
):
    user_ratings = df_user[df_user["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= args.liked_threshold]
    if liked.empty:
        return None

    holdout_count = max(1, math.ceil(len(liked) * args.test_fraction))
    if len(user_ratings) - holdout_count < args.min_train_ratings:
        return None

    holdout_indices = rng.choice(liked.index.to_numpy(), size=holdout_count, replace=False)
    heldout_titles = set(df_user.loc[holdout_indices, "title"])
    train_df = df_user.drop(index=holdout_indices)

    recommendations = hybrid_recommendations(
        user_id,
        train_df,
        df_content,
        content_similarity,
        title_to_index,
        args.k,
    )
    recommended_titles = set(recommendations)
    hits = len(recommended_titles.intersection(heldout_titles))

    return {
        "user_id": int(user_id),
        "heldout": len(heldout_titles),
        "recommended": len(recommendations),
        "hits": hits,
        f"precision@{args.k}": hits / args.k,
        f"recall@{args.k}": hits / len(heldout_titles),
        f"hit_rate@{args.k}": 1.0 if hits else 0.0,
    }


def main():
    args = parse_args()
    df_content, df_user, content_similarity = load_data()
    title_to_index = {title: index for index, title in enumerate(df_content["title"])}
    rng = np.random.default_rng(args.random_state)

    eligible_users = df_user.groupby("user_id").size()
    eligible_users = eligible_users[eligible_users >= args.min_train_ratings].index.to_list()
    rng.shuffle(eligible_users)
    if args.max_users:
        eligible_users = eligible_users[: args.max_users]

    rows = []
    for user_id in eligible_users:
        result = evaluate_user(
            user_id,
            df_user,
            df_content,
            content_similarity,
            title_to_index,
            args,
            rng,
        )
        if result:
            rows.append(result)

    if not rows:
        raise SystemExit("No users could be evaluated with the selected settings.")

    results = pd.DataFrame(rows)
    precision_col = f"precision@{args.k}"
    recall_col = f"recall@{args.k}"
    hit_rate_col = f"hit_rate@{args.k}"

    print(f"Evaluated users: {len(results):,}")
    print(f"Average {precision_col}: {results[precision_col].mean():.4f}")
    print(f"Average {recall_col}: {results[recall_col].mean():.4f}")
    print(f"Average {hit_rate_col}: {results[hit_rate_col].mean():.4f}")
    print(f"Total hits: {int(results['hits'].sum()):,}")
    print(f"Total held-out liked movies: {int(results['heldout'].sum()):,}")


if __name__ == "__main__":
    main()
