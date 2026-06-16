# ETL Project

This repository contains end-to-end ETL pipelines for the Iris and Titanic datasets, including extraction, transformation, and loading to Supabase.

## Project Structure
- `.env` – Supabase credentials (URL/Key).
- `etl_analysis_iris.ipynb` – Exploratory analysis of the Iris dataset.
- `data/`
  - `raw/`
    - `iris_data.csv` – Extracted raw Iris data.
    - `titanic_data.csv` – Extracted raw Titanic data.
  - `staged/`
    - `iris_transformed.csv` – Transformed Iris data ready for loading.
    - `titanic_transformed.csv` – Transformed Titanic data ready for loading.
- `scripts/`
  - [`extract_iris.py`](scripts/extract_iris.py) – Downloads the Iris dataset via seaborn and saves it to `data/raw/iris_data.csv`.
  - [`transform_iris.py`](scripts/transform_iris.py) – Cleans/enriches Iris data and writes `data/staged/iris_transformed.csv`.
  - [`load_iris.py`](scripts/load_iris.py) – Creates the `iris_data` table (if needed) and loads staged Iris data to Supabase.
  - [`extract_titanic.py`](scripts/extract_titanic.py) – Downloads the Titanic dataset via seaborn and saves it to `data/raw/titanic_data.csv`.
  - [`transform_titanic.py`](scripts/transform_titanic.py) – Cleans/enriches Titanic data (handles missing values, feature engineering) and writes `data/staged/titanic_transformed.csv`.
  - [`load_titanic.py`](scripts/load_titanic.py) – Creates the `titanic_data` table (if needed) and loads staged Titanic data to Supabase.

## Prerequisites
- Python 3.8+
- pip
- Supabase project (URL and service key)

## Environment Variables
Set in `.env`:
- `SUPABASE_URL`
- `SUPABASE_KEY`

## Setup
1. Install dependencies:
   ```bash
   pip install pandas seaborn supabase python-dotenv
   ```
2. Ensure `.env` contains your Supabase credentials.

## Running the Pipelines

### Iris
1. Extract raw data:
   ```bash
   python scripts/extract_iris.py
   ```
2. Transform data:
   ```bash
   python scripts/transform_iris.py
   ```
3. Load to Supabase:
   ```bash
   python scripts/load_iris.py
   ```

### Titanic
1. Extract raw data:
   ```bash
   python scripts/extract_titanic.py
   ```
2. Transform data:
   ```bash
   python scripts/transform_titanic.py
   ```
3. Load to Supabase:
   ```bash
   python scripts/load_titanic.py
   ```

## Notes
- Transformation outputs are written to `data/staged/`.
- Loading scripts batch-insert rows into Supabase and ensure target tables exist.
- The notebooks in `etl_analysis_iris.ipynb` showcase exploratory analysis on the transformed Iris dataset.