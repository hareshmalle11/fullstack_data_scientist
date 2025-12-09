import pandas as pd
import os
from supabase import create_client, Client
from dotenv import load_dotenv

# initialize supabase client
def get_supabase_client():
    load_dotenv()
    url=os.getenv("SUPABASE_URL")
    key=os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Supabase URL or Key not found in environment variables")
   
    return create_client(url,key)
def create_table_if_not_exists():
    try:
        supabase=get_supabase_client()
        create_table_sql="create table if not exists iris_data (id BIGSERIAL PRIMARY KEY,sepal_length float, sepal_width float, petal_length float, petal_width float, species text, sepal_ratio float, petal_ratio float, is_petal_long boolean);"
        try:
            supabase.rpc("execute_sql",{"query":create_table_sql}).execute()
            print("Table 'iris_data' ensured to exist.")
        except Exception as e:
            print(f"Error executing SQL to create table: {e}")
    except Exception as e:
        print(f"Error creating Supabase client: {e}")
        return
def load_to_supabase(staged_path:str,table_name:str="iris_data"):
    if not os.path.isabs(staged_path):
        staged_path=os.path.abspath(os.path.join(os.path.dirname(__file__),staged_path))
        print("looking for data file at staged path:",staged_path)
    if not os.path.exists(staged_path):
        print(f"Staged file not found at {staged_path}. Please run the transform script first.")
    try:
        supabase=get_supabase_client()
        df=pd.read_csv(staged_path)
        total_rows=len(df)
        batch_size=50
        print(f"Loading {total_rows} rows into Supabase table '{table_name}' in batches of {batch_size}...")
        # insert data in batches
        for i in range(0,total_rows,batch_size):
            batch=df.iloc[i:i+batch_size].copy()
            batch=batch.where(pd.notnull(batch),None)
            response=supabase.table(table_name).insert(batch.to_dict(orient='records')).execute()
            print(f"Inserted batch rows {i} to {min(i+batch_size-1, total_rows-1)}; response: {response}")

    except Exception as e:
        print(f"Error loading data to Supabase: {e}")
        return
if __name__ == "__main__":
    base_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    staged_path=os.path.join(base_dir,'data','staged','iris_transformed.csv')
    create_table_if_not_exists()
    load_to_supabase(staged_path)
