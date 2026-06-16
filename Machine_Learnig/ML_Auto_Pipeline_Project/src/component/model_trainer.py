import time
import importlib
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import itertools

try:
    _xgb_module = importlib.import_module("xgboost")
    XGBRegressor = getattr(_xgb_module, "XGBRegressor")
except Exception:
    XGBRegressor = None

try:
    _xgb_module = importlib.import_module("xgboost")
    XGBClassifier = getattr(_xgb_module, "XGBClassifier")
except Exception:
    XGBClassifier = None


ModelInput = Union[Type[BaseEstimator], BaseEstimator]


class ModelTrainer:
    """
    Utility class to train and evaluate ML models for regression/classification tasks.
    """

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        problem_type: str = "regression",
    ) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.problem_type = problem_type.lower()

    @staticmethod
    def get_regression_models() -> Dict[str, Type[BaseEstimator]]:
        """
        Registry of supported regression models.
        """
        models: Dict[str, Type[BaseEstimator]] = {
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "SVR": SVR,
            "KNeighborsRegressor": KNeighborsRegressor,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "Ridge": Ridge,
            "Lasso": Lasso,
        }

        if XGBRegressor is not None:
            models["XGBRegressor"] = XGBRegressor

        return models

    @staticmethod
    def _build_model(model: ModelInput, parameters: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        params = parameters or {}
        model_cls = model if isinstance(model, type) else model.__class__
        return model_cls(**params)

    def _score(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
        y_pred = model.predict(X)
        if self.problem_type == "classification":
            return float(accuracy_score(y, y_pred))
        return float(r2_score(y, y_pred))

    def train_single_model(
        self,
        model: ModelInput,
        parameters: Optional[Dict[str, Any]] = None,
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train one model and return structured metrics.
        """
        start = time.perf_counter()
        estimator = self._build_model(model, parameters)
        estimator.fit(self.X_train, self.y_train)

        train_score = self._score(estimator, self.X_train, self.y_train)
        test_score = self._score(estimator, self.X_test, self.y_test)

        cv_score = None
        if cv_folds is not None and cv_folds > 1:
            score_values = cross_val_score(
                estimator,
                self.X_train,
                self.y_train,
                cv=cv_folds,
                scoring=scoring,
            )
            cv_score = float(np.mean(score_values))

        elapsed = time.perf_counter() - start

        resolved_name = model_name or estimator.__class__.__name__
        return {
            "model_name": resolved_name,
            "parameters_used": parameters or {},
            "parameters": parameters or {},
            "train_score": float(train_score),
            "test_score": float(test_score),
            "cv_score": cv_score,
            "training_time": float(elapsed),
            "Model": resolved_name,
            "Parameters": parameters or {},
            "Train Score": float(train_score),
            "Test Score": float(test_score),
            "CV Score": cv_score,
            "Training Time": float(elapsed),
        }

    def train_with_parameters(
        self,
        model_class: Type[BaseEstimator],
        parameters: Optional[Dict[str, Any]],
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train a model class with user-provided parameters.
        """
        return self.train_single_model(
            model=model_class,
            parameters=parameters,
            cv_folds=cv_folds,
            scoring=scoring,
            model_name=model_name,
        )

    def train_selected_models(
        self,
        selected_models: Dict[str, Union[ModelInput, Dict[str, Any], tuple]],
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Train multiple selected models and return a list of structured results.

        accepted formats in selected_models values:
        - model_class_or_object
        - {"model": model_class_or_object, "parameters": {...}}
        - (model_class_or_object, {"param": value})
        """
        results: List[Dict[str, Any]] = []

        for name, model_config in selected_models.items():
            parameters: Dict[str, Any] = {}

            if isinstance(model_config, tuple):
                model_item = model_config[0]
                if len(model_config) > 1 and isinstance(model_config[1], dict):
                    parameters = model_config[1]
            elif isinstance(model_config, dict) and "model" in model_config:
                model_item = model_config["model"]
                parameters = model_config.get("parameters", {})
            else:
                model_item = model_config

            result = self.train_single_model(
                model=model_item,
                parameters=parameters,
                cv_folds=cv_folds,
                scoring=scoring,
                model_name=name,
            )
            results.append(result)

        return results

    def run_grid_search(
        self,
        model_class: Type[BaseEstimator],
        parameter_grid: Dict[str, Any],
        cv_folds: int = 5,
        scoring: Optional[str] = None,
        n_jobs: int = -1,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run GridSearchCV and return best params, score, and full results table.
        """
        start = time.perf_counter()
        grid = GridSearchCV(
            estimator=model_class(),
            param_grid=parameter_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=True,
        )

        grid.fit(self.X_train, self.y_train)
        elapsed = time.perf_counter() - start

        cv_results = pd.DataFrame(grid.cv_results_)
        columns_to_keep = [
            "params",
            "mean_train_score",
            "mean_test_score",
            "rank_test_score",
            "mean_fit_time",
        ]
        available_columns = [c for c in columns_to_keep if c in cv_results.columns]
        results_table = cv_results[available_columns].copy()
        results_table.rename(
            columns={
                "params": "Parameters",
                "mean_train_score": "Train Score",
                "mean_test_score": "Test Score",
                "rank_test_score": "Rank",
                "mean_fit_time": "Training Time",
            },
            inplace=True,
        )

        best_name = model_name or model_class.__name__
        results_table["Model"] = best_name
        return {
            "model_name": best_name,
            "best_parameters": grid.best_params_,
            "best_score": float(grid.best_score_),
            "training_time": float(elapsed),
            "full_results_table": results_table,
        }

    @staticmethod
    def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert training results to a consistent experiment table.
        """
        rows = []
        for item in results:
            rows.append(
                {
                    "Model": item.get("model_name", item.get("Model")),
                    "Parameters": item.get("parameters_used", item.get("Parameters", {})),
                    "Train Score": item.get("train_score", item.get("Train Score")),
                    "Test Score": item.get("test_score", item.get("Test Score")),
                    "CV Score": item.get("cv_score", item.get("CV Score")),
                    "Training Time": item.get("training_time", item.get("Training Time")),
                }
            )
        return pd.DataFrame(rows)

class ClassificationTrainer(ModelTrainer):
    """
    Extends ModelTrainer specifically for handling Classification tasks.
    """
    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        super().__init__(X_train, X_test, y_train, y_test, problem_type="classification")

    @staticmethod
    def get_classification_models() -> Dict[str, Type[BaseEstimator]]:
        """
        Registry of supported classification models.
        """
        models: Dict[str, Type[BaseEstimator]] = {
            "RandomForestClassifier": RandomForestClassifier,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "KNeighborsClassifier": KNeighborsClassifier,
            "DecisionTreeClassifier": DecisionTreeClassifier,
        }

        if XGBClassifier is not None:
            models["XGBClassifier"] = XGBClassifier

        return models

    def _score(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Returns primary metric for fallback (Accuracy).
        """
        y_pred = model.predict(X)
        return float(accuracy_score(y, y_pred))

    def train_single_model(
        self,
        model: ModelInput,
        parameters: Optional[Dict[str, Any]] = None,
        cv_folds: Optional[int] = None,
        scoring: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train one model and return classification metrics (Accuracy, F1, Precision, Recall).
        """
        start = time.perf_counter()
        estimator = self._build_model(model, parameters)
        estimator.fit(self.X_train, self.y_train)

        y_test_pred = estimator.predict(self.X_test)
        
        # In case of multiclass, use weighted average
        is_binary = len(np.unique(self.y_test)) == 2
        avg_method = "binary" if is_binary else "weighted"

        test_acc = accuracy_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred, average=avg_method)
        test_precision = precision_score(self.y_test, y_test_pred, average=avg_method, zero_division=0)
        test_recall = recall_score(self.y_test, y_test_pred, average=avg_method, zero_division=0)
        
        y_train_pred = estimator.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, y_train_pred)

        cv_score = None
        if cv_folds is not None and cv_folds > 1:
            score_values = cross_val_score(
                estimator,
                self.X_train,
                self.y_train,
                cv=cv_folds,
                scoring=scoring if scoring else "accuracy",
            )
            cv_score = float(np.mean(score_values))

        elapsed = time.perf_counter() - start

        resolved_name = model_name or estimator.__class__.__name__
        return {
            "model_name": resolved_name,
            "parameters_used": parameters or {},
            "parameters": parameters or {},
            "train_score": float(train_acc),
            "test_score": float(test_acc), # for backwards compat sorting
            "test_f1": float(test_f1),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "cv_score": cv_score,
            "training_time": float(elapsed),
            "Model": resolved_name,
            "Parameters": parameters or {},
            "Train Accuracy": float(train_acc),
            "Test Accuracy": float(test_acc),
            "Test F1": float(test_f1),
            "CV Score": cv_score,
            "Training Time": float(elapsed),
        }

    @staticmethod
    def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert training results to a classification experiment table.
        """
        rows = []
        for item in results:
            rows.append(
                {
                    "Model": item.get("Model", item.get("model_name")),
                    "Parameters": item.get("Parameters", item.get("parameters_used", {})),
                    "Train Accuracy": item.get("Train Accuracy", item.get("train_score")),
                    "Test Accuracy": item.get("Test Accuracy", item.get("test_score")),
                    "Test F1": item.get("Test F1", item.get("test_f1")),
                    "CV Score": item.get("CV Score", item.get("cv_score")),
                    "Training Time": item.get("Training Time", item.get("training_time")),
                }
            )
        return pd.DataFrame(rows)

class ClusteringTrainer:
    """
    Trainer for Unsupervised Learning (Clustering) tasks.
    We don't inherit from ModelTrainer because the signature and logic for unsupervised is fundamentally different (no y).
    """
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
        self.X_train = X_train
        self.X_test = X_test

    @staticmethod
    def get_clustering_models() -> Dict[str, Type[BaseEstimator]]:
        return {
            "KMeans": KMeans,
            "DBSCAN": DBSCAN,
            "AgglomerativeClustering": AgglomerativeClustering,
        }

    def train_selected_models(
        self,
        selected_models: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        results = []
        for name, config_dict in selected_models.items():
            model_cls = config_dict["model"]
            params = config_dict.get("parameters", {})
            start = time.perf_counter()
            
            estimator = model_cls(**params)
            # fit_predict to get labels on train set
            train_labels = estimator.fit_predict(self.X_train)
            
            # evaluate train
            if len(set(train_labels)) > 1:
                train_silhouette = float(silhouette_score(self.X_train, train_labels))
                train_db = float(davies_bouldin_score(self.X_train, train_labels))
            else:
                train_silhouette = 0.0
                train_db = 0.0
                
            # predict test
            if hasattr(estimator, "predict"):
                test_labels = estimator.predict(self.X_test)
            else:
                # Fallback for DBSCAN, AgglomerativeClustering using KNN 1
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(self.X_train, train_labels)
                test_labels = knn.predict(self.X_test)
                
            if len(set(test_labels)) > 1:
                test_silhouette = float(silhouette_score(self.X_test, test_labels))
                test_db = float(davies_bouldin_score(self.X_test, test_labels))
            else:
                test_silhouette = 0.0
                test_db = 0.0
                
            elapsed = time.perf_counter() - start
            results.append({
                "Model": name,
                "model_name": name,
                "Parameters": params,
                "parameters": params,
                "Train Silhouette": train_silhouette,
                "Test Silhouette": test_silhouette,
                "Train Davies-Bouldin": train_db,
                "Test Davies-Bouldin": test_db,
                "Training Time": float(elapsed),
                "model_obj": estimator
            })
        return results

    def run_grid_search(
        self,
        model_class: Type[BaseEstimator],
        parameter_grid: Dict[str, Any],
        model_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Manually run grid search since GridSearchCV requires y.
        """
        keys, values = zip(*parameter_grid.items()) if parameter_grid else ([], [])
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        if not permutations:
            permutations = [{}]

        best_score = -1.0 # Silhouette ranges from -1 to 1
        best_params = {}
        all_results = []
        
        start = time.perf_counter()
        
        for p in permutations:
            estimator = model_class(**p)
            train_labels = estimator.fit_predict(self.X_train)
            if len(set(train_labels)) > 1:
                score = float(silhouette_score(self.X_train, train_labels))
            else:
                score = -1.0
                
            if hasattr(estimator, "predict"):
                test_labels = estimator.predict(self.X_test)
            else:
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(self.X_train, train_labels)
                test_labels = knn.predict(self.X_test)

            if len(set(test_labels)) > 1:
                t_score = float(silhouette_score(self.X_test, test_labels))
            else:
                t_score = -1.0

            if score > best_score:
                best_score = score
                best_params = p
                
            all_results.append({
                "Parameters": p,
                "Train Silhouette": score,
                "Test Silhouette": t_score
            })
            
        elapsed = time.perf_counter() - start
        results_df = pd.DataFrame(all_results)
        results_df["Model"] = model_name
        
        return {
            "model_name": model_name,
            "best_parameters": best_params,
            "best_score": best_score,
            "training_time": float(elapsed),
            "full_results_table": results_df,
        }

    @staticmethod
    def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for item in results:
            rows.append({
                "Model": item.get("Model"),
                "Parameters": item.get("Parameters", {}),
                "Train Silhouette": item.get("Train Silhouette", 0.0),
                "Test Silhouette": item.get("Test Silhouette", 0.0),
                "Train DB-Index": item.get("Train Davies-Bouldin", 0.0),
                "Test DB-Index": item.get("Test Davies-Bouldin", 0.0),
                "Training Time": item.get("Training Time", 0.0)
            })
        return pd.DataFrame(rows)
