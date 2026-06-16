import os
import pandas as pd
from src.utils import load_object


class PredictionPipeline:
	"""
	Load a saved best-model artifact and generate predictions.
	"""

	def __init__(self, artifact_path: str):
		if not os.path.exists(artifact_path):
			raise FileNotFoundError(f"Model artifact not found: {artifact_path}")
		self.artifact_path = artifact_path
		self.bundle = load_object(artifact_path)
		self.model = self.bundle["model"]
		self.feature_columns = self.bundle.get("feature_columns", [])
		self.target_column = self.bundle.get("target_column")
		self.model_name = self.bundle.get("model_name", "SavedModel")
		self.label_encoder = self.bundle.get("label_encoder", None)

	def _prepare_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
		feature_df = input_df.copy()

		if self.target_column in feature_df.columns:
			feature_df = feature_df.drop(columns=[self.target_column])

		encoded = pd.get_dummies(feature_df, drop_first=False)
		aligned = encoded.reindex(columns=self.feature_columns, fill_value=0)
		return aligned

	def predict(self, input_df: pd.DataFrame):
		prepared_X = self._prepare_features(input_df)
		predictions = self.model.predict(prepared_X)
		if self.label_encoder is not None:
			predictions = self.label_encoder.inverse_transform(predictions)
		return predictions

	def predict_from_dict(self, input_data: dict):
		input_df = pd.DataFrame([input_data])
		return self.predict(input_df)
