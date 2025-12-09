import time
import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv
load_dotenv()
supabase=create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
def load_weather_data():
    # load transformed data
    staged_file="../data/staged/transformed_weather.csv"
    if not os.path.exists(staged_file):
        raise FileNotFoundError(f"{staged_file} not found. Please run the transform step first.")
    df=pd.read_csv(staged_file)
    # convert time stampt into strings
    df["time"]=pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df["extracted_at"]=pd.to_datetime(df["extracted_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    batch_size=20
    for i in range(0, len(df), batch_size):
        batch=df.iloc[i:i+batch_size].where(pd.notnull(df), None).to_dict("records")
    values = []

    for r in batch:
        values.append(
            f"('{r['time']}', {r.get('temperature', 'NULL')}, {r.get('humidity', 'NULL')}, "
            f"{r.get('wind_speed', 'NULL')}, '{r.get('city','Hyderabad')}', '{r['extracted_at']}')"
        )
    insert_query=f"""
    INSERT INTO weather_data (time, temperature, humidity, wind_speed, city, extracted_at)
    VALUES {','.join(values)}

    """


    response=supabase.rpc("execute_sql", {"query": insert_query}).execute()
    print(f"Inserted batch {i//batch_size + 1}: {response}")
    time.sleep(0.5)  # to avoid overwhelming the database

        

    
if __name__ == "__main__":
    load_weather_data()
