# hubmap-inventory
Python package that creates an inventory for HuBMAP datasets.

The inventory is composed of three files

* a TSV with all file level features
* a JSON file with basic metadata information and file manifest
* a compressed JSON file
## About the JSON file
The JSON file is a dictionary style structure with dataset and file level information. The keys of this dictionary are

* 'data_type' - CODEX, AF, etc.
* 'directory' - directory path on Hive
* 'doi_url' - the DOI URL, if any
* 'frequencies' - frequencies of file extensions in this dataset. Useful for building histograms
* 'hubmap_id' - dataset HuBMAP ID
* 'is_protected' - True if protected, False otherwise
* 'manifest' - a dictionary with file level statistics for each file in this dataset
* 'number_of_files'
* 'pretty_size' - an easy to read string representing the size of the data directory
* 'size' - size in bytes of the data directory
* 'status' - Published, etc.
* 'uuid' - dataset UUID

The `manifest` key in the dictionary above is a list of dictionaries as well. Each dictionary has file level information about a file in the dataset. The list as a long as there are files in the dataset. The keys of each dictionary in the list are

* 'download_url' - Globus direct download URL. Does not apply for protected datasets.
* 'extension'
* 'filename'
* 'filetype' - image, sequence or other
* 'fullpath'
* 'md5'
* 'mime-type'
* 'modification_time'
* 'sha256'
* 'size' - size in bytes

## Examples
See `examples` folder for Jupyter Notebooks and simple scripts.

---
Copyright © 2020-2023 HuBMAP.