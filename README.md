# Movie Recommendation System

A personalized movie recommendation web app built with Streamlit, MySQL, MovieLens ratings, IMDb/TMDB metadata, and a hybrid recommendation approach.

The app lets users create an account, save movie ratings, generate recommendations from their full rating history, or generate a one-time recommendation from a selected group of movies. Recommended movies can display posters from the TMDB API.

## Features

- User signup, login, and sign out
- Secure password hashing with PBKDF2-SHA256
- MySQL-backed users, ratings, poster data, and recommendation history
- Hybrid recommendations using content similarity and user-based collaborative filtering
- Two recommendation modes:
  - Saved history: uses all ratings saved by the signed-in user
  - Current selection only: uses a temporary set of selected movies
- Optional saving of current selections to user history
- Lazy TMDB poster fetching for recommended movies
- CSV fallback for recommender data loading when database data is unavailable

## Tech Stack

- Python
- Streamlit
- MySQL
- SQLAlchemy
- PyMySQL
- pandas, NumPy, SciPy, scikit-learn
- TMDB API

## Project Structure

```text
.
├── recommender_app/
│   ├── app.py
│   ├── database.py
│   ├── clean_content.csv
│   └── ratings_title.csv
├── scripts/
│   ├── build_similarity_matrix.py
│   ├── init_mysql.py
│   └── fetch_posters.py
├── .streamlit/
│   └── secrets.example.toml
├── DATABASE_SETUP.md
├── requirements.txt
└── README.md
```

## Recommendation Approach

This project combines two recommendation strategies:

- Content-based filtering compares movies using movie metadata and text-derived features.
- User-based collaborative filtering compares users through historical movie ratings.

The final ranked recommendations combine both signals so the system can use a user's own movie preferences while still benefiting from patterns in ratings from other users.

## Local Setup

Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Generate the local similarity matrix:

```powershell
python scripts\build_similarity_matrix.py
```

This creates `recommender_app/movie_similarity_matrix.pkl`. The file is intentionally not committed because it is large.

Create your local secrets file:

```powershell
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
```

Update `.streamlit/secrets.toml` with your local MySQL connection URL and TMDB API key.

Set up MySQL and import the starter data:

```powershell
python scripts\init_mysql.py --replace
```

Run the app:

```powershell
streamlit run recommender_app\app.py
```

The app should open at a local URL like `http://localhost:8501`.

## MySQL Setup

See [DATABASE_SETUP.md](DATABASE_SETUP.md) for the full database setup, import, and poster-fetching steps.

## TMDB Posters

The app fetches posters from TMDB only when recommendations are generated and a recommended movie is missing a stored poster URL. This keeps startup fast and avoids calling the API for every movie in the dataset.

You can also prefill poster data in batches:

```powershell
python scripts\fetch_posters.py --limit 250
```

## Secrets

Do not commit real credentials.

The real file below is intentionally ignored:

```text
.streamlit/secrets.toml
```

Only commit the safe template:

```text
.streamlit/secrets.example.toml
```

The generated similarity matrix is also ignored:

```text
recommender_app/movie_similarity_matrix.pkl
```

Regenerate it locally with:

```powershell
python scripts\build_similarity_matrix.py
```

## Deployment Notes

Push the project to GitHub first, then deploy from GitHub.

For deployment, `localhost` MySQL will not work because the deployed app runs on a remote server. Use a hosted MySQL database, then add these values in the deployment platform's secrets/settings:

```toml
database_url = "mysql+pymysql://username:password@host:3306/movie_recommender"
tmdb_api_key = "your_tmdb_api_key"
```

For Streamlit Community Cloud, set the app entry point to:

```text
recommender_app/app.py
```

## Resume Highlights

- Built an authenticated recommendation app with persistent user rating history
- Integrated MySQL for users, ratings, poster metadata, and recommendation logs
- Implemented hybrid recommendation logic using content similarity and collaborative filtering
- Added TMDB API integration with lazy poster fetching and database caching
- Used Streamlit for an interactive end-to-end machine learning application
