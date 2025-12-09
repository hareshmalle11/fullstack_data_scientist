# ETL Live API Pipeline

End-to-end ETL that fetches NASA APOD data and Open-Meteo weather data, stages them as CSV, and loads them into Supabase tables.

## Project Structure

```
ETL_live_api/
├─ README.md
├─ .env                      # SUPABASE_URL, SUPABASE_KEY (and optional API keys)
├─ requirements.txt
├─ data/
│  ├─ raw/                   # Unprocessed API JSON
│  │  ├─ nasa.json
│  │  └─ weather_data_YYYYMMDD_HHMMSS.json
│  └─ staged/                # Transformed CSVs ready to load
│     ├─ transformed_nasa.csv
│     └─ transformed_weather.csv
├─ notebooks/                # (optional) exploratory notebooks
├─ scripts/
│  ├─ extract_nasa.py        # Calls NASA APOD API, saves raw JSON
│  ├─ extract_weather.py     # Calls Open-Meteo API, saves raw JSON
│  ├─ transform_nasa.py      # Converts nasa.json → transformed_nasa.csv
│  ├─ trasnform_weather.py   # Converts weather JSON → transformed_weather.csv
│  ├─ load_nasa.py           # Loads transformed_nasa.csv → Supabase table nasa_data
│  └─ load_weather.py        # Loads transformed_weather.csv → Supabase table weather_data
└─ utils/ (optional)         # Shared helpers (logging, validation, etc.)
```

## Prerequisites

- Python 3.9+
- Install dependencies:
  ```sh
  pip install -r requirements.txt
  ```
  Ensure `requests`, `pandas`, `python-dotenv`, and the Supabase Python client are present.

- Supabase project with RPC `execute_sql` enabled for inserts.

## Configuration

Create `.env` with:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_service_role_key
```
(Ensure the service role key or a key with rights to call `execute_sql`.)

## Workflow

1) **Extract**
- NASA APOD:
  ```sh
  python scripts/extract_nasa.py
  ```
  Saves to `data/raw/nasa.json`.
- Weather:
  ```sh
  python scripts/extract_weather.py
  ```
  Saves timestamped JSON to `data/raw/`.

2) **Transform**
- NASA:
  ```sh
  python scripts/transform_nasa.py
  ```
  Outputs `data/staged/transformed_nasa.csv`.
- Weather:
  ```sh
  python scripts/trasnform_weather.py
  ```
  Outputs `data/staged/transformed_weather.csv`.

3) **Load**
- NASA → `nasa_data`:
  ```sh
  python scripts/load_nasa.py
  ```
- Weather → `weather_data`:
  ```sh
  python scripts/load_weather.py
  ```

## Notes & Tips

- The loaders use `supabase.rpc("execute_sql", ...)`; ensure the RPC exists and the caller is authorized.
- Basic throttling (`time.sleep`) is applied per row to avoid rate/DB pressure.
- Weather defaults (lat `17.3850`, lon `78.4867`, `forecast_days=1`) can be tweaked in `scripts/extract_weather.py`.
- Keep raw and staged data under `data/` out of version control if they contain sensitive info (update `.gitignore` accordingly).