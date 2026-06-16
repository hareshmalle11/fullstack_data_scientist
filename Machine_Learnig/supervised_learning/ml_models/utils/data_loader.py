import pandas as pd

def load_csv(file):
    """
    Load CSV file into a pandas DataFrame with error handling
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")

def validate_dataframe(df):
    """
    Validate that dataframe meets basic requirements
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty")
    
    if df.shape[0] < 10: 
        raise ValueError("Dataset should have at least 10 rows")
    
    return True