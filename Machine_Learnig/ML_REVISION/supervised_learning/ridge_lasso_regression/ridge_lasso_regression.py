"""Train Ridge and Lasso regression models for car price prediction."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "CarPrice_Assignment.csv"
MODEL_PATH = BASE_DIR / "ridge_lasso_model.pkl"


def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(columns=["car_ID", "CarName"])
    return data


def build_pipeline(model):
    data = load_data()
    feature_columns = [column for column in data.columns if column != "price"]
    numeric_features = data[feature_columns].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = data[feature_columns].select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_models():
    data = load_data()
    X = data.drop(columns=["price"])
    y = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "Ridge Regression": {
            "pipeline": build_pipeline(Ridge(random_state=42)),
            "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "Lasso Regression": {
            "pipeline": build_pipeline(Lasso(random_state=42, max_iter=100000)),
            "params": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
    }

    results = {}
    best_name = None
    best_search = None
    best_score = float("-inf")

    for name, config in candidates.items():
        search = GridSearchCV(
            config["pipeline"],
            config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        predictions = search.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "best_alpha": search.best_params_["model__alpha"],
            "cv_r2": search.best_score_,
            "test_r2": r2_score(y_test, predictions),
            "mae": mean_absolute_error(y_test, predictions),
            "mse": mse,
            "rmse": mse ** 0.5,
        }
        results[name] = metrics

        if metrics["test_r2"] > best_score:
            best_score = metrics["test_r2"]
            best_name = name
            best_search = search

    bundle = {
        "model": best_search.best_estimator_,
        "best_model_name": best_name,
        "metrics": results,
        "feature_columns": X.columns.tolist(),
        "input_defaults": X.median(numeric_only=True).to_dict(),
        "category_options": {
            column: sorted(X[column].dropna().unique().tolist())
            for column in X.select_dtypes(include=["object"]).columns
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    model_bundle = train_models()
    print(f"Saved best model: {model_bundle['best_model_name']}")
    for model_name, metrics in model_bundle["metrics"].items():
        print(
            f"{model_name}: alpha={metrics['best_alpha']}, "
            f"R2={metrics['test_r2']:.3f}, RMSE={metrics['rmse']:.2f}, "
            f"MAE={metrics['mae']:.2f}"
        )
