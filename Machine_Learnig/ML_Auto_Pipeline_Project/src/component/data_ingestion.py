import os
import pandas as pd
from src.logger import logging

ARTIFACTS_DIR = os.path.join(os.getcwd(), "artifacts")


class DataIngestion:
    def ingest(self, uploaded_file) -> pd.DataFrame:
        """
        Read an uploaded CSV or Excel file, save raw copy to
        artifacts/raw_data.csv, and return as a DataFrame.
        """
        logging.info("DataIngestion: starting file ingestion")

        os.makedirs(ARTIFACTS_DIR, exist_ok=True)

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        raw_path = os.path.join(ARTIFACTS_DIR, "raw_data.csv")
        df.to_csv(raw_path, index=False)

        logging.info(
            f"DataIngestion: saved raw data to {raw_path} "
            f"— shape {df.shape}"
        )
        return df
