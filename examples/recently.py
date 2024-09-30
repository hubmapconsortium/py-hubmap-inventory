
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

token = os.getenv("TOKEN")
if token is None:
    print("Error: TOKEN environment variable is not set")
    sys.exit(1)

df = hubmapbags.reports.daily()

df["published_date"] = df["published_timestamp"].apply(_convert_to_datetime)
df["published_date"] = pd.to_datetime(df["published_date"])
df = df[df["published_date"].dt.year == 2024]
df = df[df["status"] == "Published"]

for index, datum in df.iterrows():
    try:
        data = hubmapinventory.inventory.create(
            datum['hubmap_id'],
            token=token,
            ncores=16,
            dbgap_study_id=None,
            compute_uuids=True,
            recompute_file_extension=False,
        )
    except:
        print(f"Failed to process dataset {value}.")
        print(traceback.format_exc())
