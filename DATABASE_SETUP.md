# MySQL and API setup

The Streamlit app uses MySQL for authentication, saved ratings, recommendation history, and poster metadata.

## 1. Create the database

In MySQL:

```sql
CREATE DATABASE movie_recommender CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'movie_app'@'localhost' IDENTIFIED BY 'change_this_password';
GRANT ALL PRIVILEGES ON movie_recommender.* TO 'movie_app'@'localhost';
FLUSH PRIVILEGES;
```

You can also use your existing MySQL user instead of creating `movie_app`, but a dedicated app user is safer.

## 2. Install dependencies

```powershell
venv\Scripts\pip.exe install -r requirements.txt
```

## 3. Configure the app

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml`, then update the URL:

```toml
database_url = "mysql+pymysql://movie_app:change_this_password@localhost:3306/movie_recommender"
```

If your password contains special characters like `@`, URL-encode them in the connection string. For example, `@` becomes `%40`.

You can also set an environment variable instead:

```powershell
$env:DATABASE_URL = "mysql+pymysql://movie_app:change_this_password@localhost:3306/movie_recommender"
```

## 4. Import the existing CSV data

```powershell
venv\Scripts\python.exe scripts\init_mysql.py --replace
```

The script creates these tables:

- `movies`
- `ratings`
- `app_users`
- `recommendation_history`

## 5. Run the app

If `recommender_app/movie_similarity_matrix.pkl` is missing, generate it first:

```powershell
venv\Scripts\python.exe scripts\build_similarity_matrix.py
```

```powershell
venv\Scripts\python.exe -m streamlit run recommender_app\app.py --server.port=8504
```

When MySQL is working, the app shows `Data source: MySQL`.

## 6. Configure TMDB posters

Create a TMDB account, generate an API key, and add it to `.streamlit/secrets.toml`:

```toml
tmdb_api_key = "your_tmdb_api_key"
```

The app fetches missing posters lazily after recommendations are generated. You can also prefill posters in batches:

```powershell
venv\Scripts\python.exe scripts\fetch_posters.py --limit 250
```

The script only fetches movies missing poster URLs, so it can be run repeatedly until all available posters are filled.

## Deployment

Push the project to GitHub before deployment.

For a deployed Streamlit app, do not use a local database URL like `localhost`. The deployed app needs a hosted MySQL database URL and the same table setup from this file.

Add these as deployment secrets:

```toml
database_url = "mysql+pymysql://username:password@host:3306/movie_recommender"
tmdb_api_key = "your_tmdb_api_key"
```

For Streamlit Community Cloud, use this entry point:

```text
recommender_app/app.py
```
