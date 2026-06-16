import json
from pathlib import Path
from datetime import datetime
import requests

# where to store raw data
DATA_DIR = Path(__file__).resolve().parents[1] / "data"/"raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
#API_KEY = ""

def extract_weather_data(lat=17.3850,lon=78.4867, days=1):
    """Extract weather data from a public API and save it as a JSON file."""
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": days,
        "timezone": "auto"
        }
    resp=requests.get(base_url, params=params)
    resp.raise_for_status()  # Raise an error for bad responses
    data= resp.json()
    filename=DATA_DIR / f"weather_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filename.write_text(json.dumps(data, indent=2))
    print(f"Weather data saved to {filename}")
    return data
if __name__ == "__main__":
    extract_weather_data()
