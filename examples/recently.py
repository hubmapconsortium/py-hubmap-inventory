import hubmapinventory
import hubmapbags
import traceback
import os
import sys
from datetime import datetime
import requests
import pandas as pd


def _convert_to_datetime(stamp):
    try:
        stamp = int(stamp) / 1000.0
        return datetime.fromtimestamp(stamp)
    except:
        return None


def fetch_data():
    url = "https://ingest.api.hubmapconsortium.org/datasets/data-status"
    try:
        # Send a GET request to the endpoint
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the JSON response
        data = response.json()
        data = data['data']

        # Convert JSON data to a DataFrame
        df = pd.DataFrame(data)

        return df
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

token = os.getenv("TOKEN")
if token is None:
    print("Error: TOKEN environment variable is not set")
    sys.exit(1)

df = fetch_data()
df["published_date"] = df["published_timestamp"].apply(_convert_to_datetime)
df["published_date"] = pd.to_datetime(df["published_date"])
df = df[df["published_date"].dt.year == 2024]
df = df[df["status"] == "Published"]

for value in df["hubmap_id"]:
    print(value)
    try:
        df = hubmapinventory.inventory.create(
            value,
            token=token,
            ncores=10,
            dbgap_study_id=None,
            compute_uuids=True,
            recompute_file_extension=False,
        )
    except:
        print(f"Failed to process dataset {value}.")
        print(traceback.format_exc())
    break
