import json
import pandas as pd
import os
import glob

def transform_weather_data():
    os.makedirs("../data/staged", exist_ok=True)
    latest_file = sorted(glob.glob("../data/raw/weather_*.json"))[-1]
    with open(latest_file, "r") as f:
        data=json.load(f)
    hourly=data["hourly"]
    df=pd.DataFrame({
        #time,temperature_2m,relative_humidity_2m,wind_speed_10m
        "time":hourly["time"],
        "temperature":hourly["temperature_2m"],
        "humidity":hourly["relative_humidity_2m"],
        "wind_speed":hourly["wind_speed_10m"]
    })
    df["city"]="Hyderabad"
    df["extracted_at"]=pd.Timestamp.now()
    output_file="../data/staged/transformed_weather.csv"
    df.to_csv(output_file, index=False)
    print(f"Transformed {len(df)} data records saved to {output_file}")
    return df
if __name__ == "__main__":
    transform_weather_data()
