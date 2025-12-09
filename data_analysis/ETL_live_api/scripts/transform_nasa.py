import json
import pandas as pd
import os
import glob
def transform_nasa_data():
    os.makedirs("../data/staged", exist_ok=True)
    raw_file="../data/raw/nasa.json"
    
    with open(raw_file, "r") as f:
        data=json.load(f)
    updated_time= pd.Timestamp.now()
    df=pd.DataFrame({
        "date":[data.get("date")],
        "title":[data.get("title")],
        
        "url":[data.get("url")],
        "media_type":[data.get("media_type")],
        "update_time":[updated_time]
    })
    output_file="../data/staged/transformed_nasa.csv"
    df.to_csv(output_file, index=False)
    print(f"Transformed NASA data saved to {output_file}")
    return df
if __name__ == "__main__":
    transform_nasa_data()