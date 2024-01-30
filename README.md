# HuBMAP Inventory Python Package

This Python package, `py-hubmap-inventory`, is designed to generate an inventory for HuBMAP datasets. The inventory consists of three files:

1. A TSV file containing all file-level features.
2. A JSON file with basic metadata information and a file manifest.
3. A compressed JSON file.

## Prerequisites

Before using this package, please note the following:

- The package requires access to the file system.
- Both protected and public published datasets can be processed on `HIVE` by the `hive` user.
- Public published datasets can be processed on `Bridges2` by any user who is part of the project, as the data is public.
- There is a processing limit associated with the maximum number of files that can be processed at once. The optimal number of cores for processing is `25`.

## JSON File Structure

The JSON file is structured as a dictionary with dataset and file-level information. The keys of this dictionary include:

- `data_type`: The type of data (e.g., CODEX, AF, etc.).
- `directory`: The directory path on Hive.
- `doi_url`: The DOI URL, if applicable.
- `frequencies`: The frequencies of file extensions in this dataset. This is useful for building histograms.
- `hubmap_id`: The dataset's HuBMAP ID.
- `is_protected`: A boolean value indicating whether the dataset is protected.
- `manifest`: A dictionary containing file-level statistics for each file in this dataset.
- `number_of_files`: The total number of files in the dataset.
- `pretty_size`: A human-readable string representing the size of the data directory.
- `size`: The size of the data directory in bytes.
- `status`: The status of the dataset (e.g., Published, etc.).
- `uuid`: The dataset's UUID.

The `manifest` key in the dictionary is a list of dictionaries, each containing file-level information about a file in the dataset. The keys of each dictionary in the list include:

- `download_url`: The Globus direct download URL. This does not apply for protected datasets.
- `extension`: The file extension.
- `filename`: The name of the file.
- `filetype`: The type of file (e.g., image, sequence, or other).
- `fullpath`: The full path to the file.
- `md5`: The file's MD5 checksum.
- `mime-type`: The file's MIME type.
- `modification_time`: The file's last modification time.
- `sha256`: The file's SHA256 checksum.
- `size`: The size of the file in bytes.

## Examples

Please refer to the `examples` folder for Jupyter Notebooks and simple scripts demonstrating how to use this package.

---

Copyright © 2020-2024 HuBMAP.