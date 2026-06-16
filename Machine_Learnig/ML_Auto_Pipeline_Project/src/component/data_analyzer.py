import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataAnalyzer:

    def get_shape(self, df: pd.DataFrame) -> dict:
        return {
            "rows": df.shape[0],
            "columns": df.shape[1]
        }

    def get_head(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        return df.head(n)

    def get_info(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.values,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Null %": (df.isnull().sum().values / len(df) * 100).round(2)
        })

    def get_describe(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.describe()
    def get_column_summary(self, df: pd.DataFrame) -> dict:
        numerical_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()
        
        unique_counts = {
            col: df[col].nunique() 
            for col in categorical_cols
        }
        
        return {
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "unique_counts": unique_counts
        }
    def get_numerical_cols(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(include="number").columns.tolist()

    def get_categorical_cols(self, df: pd.DataFrame) -> list:
        return df.select_dtypes(include="object").columns.tolist()   
       
    def get_heatmap(self, df: pd.DataFrame, col1: str, col2: str):


        corr_data = df[[col1, col2]].corr()
        
        fig, ax = plt.subplots()
        sns.heatmap(corr_data, annot=True, cmap="coolwarm", ax=ax)
        
        return fig
    def plot_histogram(self, df: pd.DataFrame, col: str):
        if col not in df.columns:
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None
        if df[col].dropna().empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 5))
        df[col].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


    def plot_boxplot(self, df: pd.DataFrame, col: str):
        if col not in df.columns:
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
            return None
        if df[col].dropna().empty:
            return None

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(y=df[col], ax=ax, color='lightgreen')
        ax.set_title(f"Box Plot: {col}")
        ax.set_ylabel(col)
        plt.tight_layout()
        return fig


    def plot_scatter(self, df: pd.DataFrame, col_x: str, col_y: str):
        if col_x not in df.columns or col_y not in df.columns:
            return None
        if col_x == col_y:
            return None
        if not (pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y])):
            return None
        if df[[col_x, col_y]].dropna().empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(df[col_x], df[col_y], alpha=0.6, s=40)
        ax.set_title(f"{col_y} vs {col_x}")
        ax.set_xlabel(col_x)
        ax.set_ylabel(col_y)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


    def plot_regression(self, df: pd.DataFrame, col_x: str, col_y: str):
        if col_x not in df.columns or col_y not in df.columns:
            return None
        if col_x == col_y:
            return None
        if not (pd.api.types.is_numeric_dtype(df[col_x]) and pd.api.types.is_numeric_dtype(df[col_y])):
            return None
        if df[[col_x, col_y]].dropna().empty:
            return None

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.regplot(x=col_x, y=col_y, data=df, ax=ax,
                    scatter_kws={'alpha':0.5, 's':40},
                    line_kws={'color':'red'})
        ax.set_title(f"{col_y} vs {col_x} (regression)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    def handle_nulls(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Fill null values using the selected method.
        Returns a new DataFrame (does not modify original).
        """
        df_clean = df.copy()

        if method == "Fill with Mean":
            for col in df_clean.select_dtypes(include=np.number).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        
        elif method == "Fill with Median":
            for col in df_clean.select_dtypes(include=np.number).columns:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        elif method == "Fill with Mode":
            for col in df_clean.columns:
                if df_clean[col].mode().size > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        return df_clean
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using IQR method.
        Returns DataFrame containing only the rows with outliers.
        """
        df_numeric = df.select_dtypes(include=np.number)
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = ((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)
        return df[outliers_mask]
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows containing outliers (IQR method).
        Returns cleaned DataFrame.
        """
        df_numeric = df.select_dtypes(include=np.number)
        Q1 = df_numeric.quantile(0.25)
        Q3 = df_numeric.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        mask = ~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)
        return df[mask].copy()
    def encode_categorical(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        Handle categorical columns:
        - "Remove Category" → drop all categorical columns
        - "Convert All into Numerical" → label encode them
        """
        df_clean = df.copy()
        
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        
        if method == "Remove Category":
            return df_clean.drop(columns=cat_cols)
        
        elif method == "Convert All into Numerical":
            le = LabelEncoder()
            for col in cat_cols:
                # Only encode if column has reasonable cardinality
                if df_clean[col].nunique() <= 50:
                    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                else:
                    # For high cardinality → drop instead of encoding
                    df_clean = df_clean.drop(columns=[col])
            return df_clean
        
        return df_clean  # fallback - return unchanged
    def remove_highly_correlated_features(self, df: pd.DataFrame, target_column: str = None, threshold: float = 0.85):

        if target_column and target_column in df.columns:
            feature_df = df.drop(columns=[target_column], errors="ignore")
        else:
            feature_df = df.copy()

        # Select numeric columns only
        num_df = feature_df.select_dtypes(include=["number"])

        if num_df.shape[1] < 2:
            return [], []

        # Compute correlation matrix
        corr_matrix = num_df.corr()

        to_drop = set()
        correlated_pairs = []

        cols = corr_matrix.columns

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):

                col1 = cols[i]
                col2 = cols[j]

                corr_val = corr_matrix.iloc[i, j]

                if abs(corr_val) >= threshold:

                    correlated_pairs.append((col1, col2, corr_val))

                    # drop second feature
                    to_drop.add(col2)

        return list(to_drop), correlated_pairs

class ClassificationAnalyzer(DataAnalyzer):
    """
    Extends DataAnalyzer for classification-specific analysis.
    """
    def check_class_imbalance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Returns class distribution counts and percentages.
        """
        if target_column not in df.columns:
            return pd.DataFrame()
        
        counts = df[target_column].value_counts()
        percentages = (df[target_column].value_counts(normalize=True) * 100).round(2)
        
        return pd.DataFrame({
            "Class": counts.index,
            "Count": counts.values,
            "Percentage (%)": percentages.values
        })
    
    def plot_class_distribution(self, df: pd.DataFrame, target_column: str):
        """
        Plots a bar chart for class distribution.
        """
        if target_column not in df.columns:
            return None
            
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=target_column, data=df, ax=ax, palette="Set2")
        ax.set_title(f"Class Distribution: {target_column}")
        ax.set_ylabel("Count")
        plt.tight_layout()
        return fig
