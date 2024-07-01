import hubmapinventory
import hubmapbags
import sys
import requests
import pandas as pd
from os import getenv
import traceback
from datetime import datetime

ncores = 10

# Get token. Needed for non-public data
token = getenv("TOKEN")
if token is None:
    print("Error: TOKEN environment variable is not set")
    sys.exit(1)

def _convert_to_datetime(stamp):
    """
    Helper method to convert timestamps to date time
    """

    try:
        stamp = int(stamp) / 1000.0
        return datetime.fromtimestamp(stamp)
    except:
        return None

def fetch_data():
    """
    Helper method that retrieves data from public endpoint
    """

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

# Get data
df = fetch_data()
#df["published_date"] = df["published_timestamp"].apply(_convert_to_datetime)
#df["published_date"] = pd.to_datetime(df["published_date"])
df = df[df["status"] == "Published"]
df = df[df['dataset_type'] == 'MUSIC']

compute_uuids = True
for index, dataset in df.iterrows():
        try:
            if (
                dataset["status"] == "Published"
                and dataset["data_access_level"] == "protected"
                and dataset["is_primary"]
            ):
                df = hubmapinventory.inventory.create(
                    dataset["hubmap_id"],
                    token=token,
                    ncores=ncores,
                    dbgap_study_id=None,
                    update_local_file=True,
                    compute_uuids=compute_uuids,
                    recompute_file_extension=True,
                )
            else:
                print(f'Avoiding computation of dataset {dataset["hubmap_id"]}.')
        except:
            print(f'Failed to process dataset {dataset["hubmap_id"]}.')
            traceback.print_exc()
