from datetime import datetime
from pathlib import Path

import hubmapbags
import pandas as pd
from tqdm import tqdm

token = "this-is-my-token"
instance = "prod"  # default instance is test

# get assay types
assay_names = hubmapbags.get_assay_types(token=token)

report = pd.DataFrame()
for assay_name in assay_names:
    print(assay_name)
    datasets = pd.DataFrame(
        hubmapbags.get_hubmap_ids(assay_name=assay_name, token=token)
    )

    if datasets.empty:
        continue

    # clean up
    datasets = datasets[(datasets["data_type"] != "image_pyramid")]
    datasets = datasets[(datasets["status"] == "Published")]

    datasets["has_uuids"] = None
    datasets["number_of_uuids"] = None
    for index, datum in tqdm(datasets.iterrows()):
        datasets.loc[index, "number_of_uuids"] = hubmapbags.uuids.get_number_of_uuids(
            datum["hubmap_id"], instance=instance, token=token
        )

        if datasets.loc[index, "number_of_uuids"] == 0:
            datasets.loc[index, "has_uuids"] = False
        else:
            datasets.loc[index, "has_uuids"] = True

    if report.empty:
        report = datasets
    else:
        report = pd.concat([report, datasets])

now = datetime.now()
directory = "uuid-data-report"

if not Path(directory).exists():
    Path(directory).mkdir()
report.to_csv(
    directory + "/" + str(now.strftime("%Y%m%d")) + ".tsv", sep="\t", index=False
)
