import pandas as pd
import os
from extract_iris import extract_data
def transform_data(raw_path):
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_dir=os.path.join(base_dir,'data','staged')
    os.makedirs(staged_dir,exist_ok=True)
    df=pd.read_csv(raw_path)
    #1. handle missing values
    numeric_cols=["sepal_length","sepal_width","petal_length","petal_width"]
    # fill missing numeric values with mean
    for col in numeric_cols:
        df[col]=df[col].fillna(df[col].mean())
        df['species']=df['species'].fillna(df['species'].mode()[0])
    # feature engineering
    df['sepal_ratio']=df['sepal_length']/df['sepal_width']
    df['petal_ratio']=df['petal_length']/df['petal_width']
    df['is_petal_long']=df['petal_length']>df['petal_length'].mean()

    # drop unnecessary columns
    df.drop(columns=[],inplace=True,errors='ignore')

    #saved transformed data
    staged_path=os.path.join(staged_dir,'iris_transformed.csv')
    df.to_csv(staged_path,index=False)
    print(f"Data transformed and saved to {staged_path}")
if __name__ == "__main__":
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path=os.path.join(base_dir,'data','raw','iris_data.csv')
    if not os.path.exists(raw_path):
        extract_data()
    transform_data(raw_path)

