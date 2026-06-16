import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split
from src.logger import logging

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")


class DataTransform:
    def scale_features(self, df: pd.DataFrame, method: str = "Standard") -> pd.DataFrame:
        """
        Scale numerical columns.
        method: 'Standard' (zero mean, unit variance) or 'MinMax' (0-1 range).
        Returns a new scaled DataFrame.
        """
        logging.info(f"DataTransform: scaling features using {method} scaler")

        df_scaled = df.copy()
        num_cols = df_scaled.select_dtypes(include="number").columns

        if method == "Standard":
            scaler = StandardScaler()
        elif method == "MinMax":
            scaler = MinMaxScaler()
        elif method == "Normalize":
            scaler = Normalizer()
        else:
            raise ValueError("Unsupported scaling method. Use Standard, MinMax, or Normalize.")

        df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])

        logging.info(f"DataTransform: scaled {len(num_cols)} numerical columns")
        return df_scaled

    def save_transformed_dataset(
        self,
        df: pd.DataFrame,
        filename: str = "transformed_data.csv"
    ) -> str:
        """
        Save transformed dataset to artifacts/ and return absolute path.
        """
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, filename)
        df.to_csv(path, index=False)
        logging.info(
            f"DataTransform: saved transformed dataset -> {path} "
            f"- shape {df.shape}"
        )
        return path

    def split_data(self, df: pd.DataFrame, target_col: str, test_size: float = 0.2):
        """
        Split df into train/test sets, save both to artifacts/, and
        return (X_train, X_test, y_train, y_test).
        """
        logging.info(
            f"DataTransform: splitting data — target={target_col}, "
            f"test_size={test_size}"
        )

        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        train_df = X_train.copy()
        train_df[target_col] = y_train.values
        test_df = X_test.copy()
        test_df[target_col] = y_test.values

        train_path = os.path.join(ARTIFACTS_DIR, "train.csv")
        test_path = os.path.join(ARTIFACTS_DIR, "test.csv")
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logging.info(
            f"DataTransform: train shape {train_df.shape}, "
            f"test shape {test_df.shape} — saved to artifacts/"
        )
        return X_train, X_test, y_train, y_test

    def save_final_dataset(self, df: pd.DataFrame, filename: str = "cleaned_data.csv") -> str:
        """
        Save the final cleaned / engineered DataFrame to artifacts/
        and return the saved file path.
        """
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        path = os.path.join(ARTIFACTS_DIR, filename)
        df.to_csv(path, index=False)
        logging.info(
            f"DataTransform: saved final dataset → {path} "
            f"— shape {df.shape}"
        )
        return path

class ClassificationTransform(DataTransform):
    """
    Extends DataTransform for classification-specific transformation,
    such as encoding the target variable.
    """
    def __init__(self):
        super().__init__()
        self.label_encoder = None

    def encode_target(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Label encodes the target column for classification.
        Saves the encoder in self.label_encoder so it can be saved with the model bundle if needed.
        """
        from sklearn.preprocessing import LabelEncoder
        
        logging.info(f"ClassificationTransform: encoding target column '{target_col}'")
        df_encoded = df.copy()
        
        if target_col in df_encoded.columns:
            if not pd.api.types.is_numeric_dtype(df_encoded[target_col]):
                self.label_encoder = LabelEncoder()
                df_encoded[target_col] = self.label_encoder.fit_transform(df_encoded[target_col].astype(str))
                logging.info(f"ClassificationTransform: target '{target_col}' encoded successfully.")
            else:
                logging.info(f"ClassificationTransform: target '{target_col}' is already numeric. Assuming it is encoded.")
        return df_encoded
