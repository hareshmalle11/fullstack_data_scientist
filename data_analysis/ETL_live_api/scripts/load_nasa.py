import time
import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()
supabase=create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
def load_nasa_data():
    # load transformed data
    staged_file="../data/staged/transformed_nasa.csv"
    if not os.path.exists(staged_file):
        raise FileNotFoundError(f"{staged_file} not found. Please run the transform step first.")
    df=pd.read_csv(staged_file)
        
    
    for _, r in df.iterrows():
        
        values = []

        
        values.append(
                f"('{r['date']}', '{r['title']}', '{r['url']}', '{r['media_type']}', '{r['update_time']}')"
     
            )
        insert_query=f"""
        INSERT INTO nasa_data (date, title, url, media_type, update_time)
        VALUES {','.join(values)}

        """


        response=supabase.rpc("execute_sql", {"query": insert_query}).execute()
        
        time.sleep(0.5)  # to avoid overwhelming the database
    print(f"Inserted {len(df)} records into nasa_data table.")
if __name__ == "__main__":
    load_nasa_data()