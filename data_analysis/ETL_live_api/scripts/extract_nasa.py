import json
from pathlib import Path
from datetime import datetime
import requests              
DATA_DIR = Path(__file__).resolve().parents[1]/"data"/"raw" #1 step above
DATA_DIR.mkdir(parents = True,exist_ok=True)
NASA_KEY='LqgWOhK8Q4gItovjYItdRI06BdIo9aTSkZhrRQgH'
def extract_nasa(api_key=NASA_KEY):
    url = "https://api.nasa.gov/planetary/apod"
    params = {
        'api_key':api_key
    }
    resp = requests.get(url=url,params=params)
    resp.raise_for_status()
    data = resp.json()
    # use a safe timestamp format (avoid %D which contains slashes on Windows)
    filename = DATA_DIR / f"nasa.json"
    # ensure parent directories exist (in case timestamp or naming introduces subfolders)
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text(json.dumps(data, indent=2))
    print(f'Extracted nasa data saved to {filename}')
if __name__=="__main__":
    extract_nasa()