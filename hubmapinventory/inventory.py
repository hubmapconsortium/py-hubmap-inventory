import datetime
import gzip
import hashlib
import json
import os
import os.path
import shutil
import typing
import uuid
import warnings
from pathlib import Path

import hubmapbags
import magic  # pyton-magic
import numpy as np
import pandas as pd
import tabulate
from numpyencoder import NumpyEncoder
from pandarallel import pandarallel
from tqdm import tqdm

try:
    from pandas.core.common import SettingWithCopyWarning

    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
except:
    warnings.filterwarnings("ignore")

###############################################################################################################
def evaluate(hubmap_id: str, token: str, debug: bool,) -> pd.DataFrame:
    """
    Returns FAIRness assessment of a particular dataset given a HuBMAP ID.
    """
    raise NotImplementedError

    # either a dataframe or a JSON block with data
    data = get(hubmap_id, token)

    # create empty container
    features = []

    # computes first feature
    features.append(__get_number_of_files(data))

    return features


# this is a metric of FAIRness
def __get_number_of_files(data):
    return None


def __get_number_of_images(data):
    return None


def __get_number_of_sequences(data):
    return None


def __get_data_type(data):
    return None


###############################################################################################################
def __pprint(msg: str):
    """
    Pretty-print a message enclosed in a horizontal line.

    This method prints the input 'msg' with a horizontal line above and below
    it to make it visually separated.

    :param msg: The message to be pretty-printed.
    :type msg: str

    :Example:

    >>> message = "Hello, Sphinx!"
    >>> __pprint(message)
    +--------------+
    |Hello, Sphinx!|
    +--------------+
    """

    row = len(msg)
    h = "".join(["+"] + ["-" * row] + ["+"])
    result = "\n" + h + "\n" "|" + msg + "|" "\n" + h
    print(result)


def __update_dataframe(
    dataset: pd.DataFrame, temp: pd.DataFrame, key: str
) -> pd.DataFrame:
    """
    Update a DataFrame with values from another DataFrame.

    :param dataset: The DataFrame to be updated.
    :type dataset: pd.DataFrame
    :param temp: The DataFrame containing new values.
    :type temp: pd.DataFrame
    :param key: The column key to use for updating the 'dataset'.
    :type key: str

    :return: The updated DataFrame.
    :rtype: pd.DataFrame

    :Example:

    >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> df2 = pd.DataFrame({'B': [7, 8, 9]})
    >>> key_column = 'B'
    >>> updated_df = __update_dataframe(df1, df2, key_column)
    >>> print(updated_df)
       A  B
    0  1  7
    1  2  8
    2  3  9
    """

    for index, datum in temp.iterrows():
        dataset.loc[index, key] = temp.loc[index, key]
    return dataset


###############################################################################################################
def get(hubmap_id: str, token: str,) -> pd.DataFrame:
    """
    Get a DataFrame from HubMap by its ID.

    This function retrieves a DataFrame from the HubMap API using the provided
    'hubmap_id' and 'token' for authentication. The DataFrame is cached locally
    to improve subsequent retrieval speed.

    :param hubmap_id: The ID of the HubMap dataset to retrieve.
    :type hubmap_id: str

    :param token: The access token for authentication with the HubMap API.
    :type token: str

    :return: The retrieved DataFrame.
    :rtype: pd.DataFrame

    :raises FileNotFoundError: If the requested file does not exist in the local cache.
    """

    metadata = hubmapbags.apis.get_dataset_info(hubmap_id, instance="prod", token=token)
    file = f'{metadata["uuid"]}.tsv'

    directory = ".data"
    filename = f"{directory}/{file}"
    if Path(filename).exists():
        return pd.read_csv(filename, sep="\t", low_memory=False)

    directory = ".data"
    filename = f"{directory}/{file}"
    if Path(filename).exists():
        return pd.read_csv(filename, sep="\t", low_memory=False)

    return pd.DataFrame()


###############################################################################################################
def create(
    hubmap_id: str,
    dbgap_study_id: str,
    token: str,
    ncores: int = 2,
    compute_uuids: bool = False,
    recompute_file_extension: bool = False,
    backup: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Main function that creates an inventory.

    :param hubmap_id: valid HuBMAP ID
    :rtype hubmap_id: string
    :param token: a valid HuBMAP token
    :rtype token: string
    :param ncores: Number of cores (number of workers)
    :rtype ncores: int
    :param compute_uuids: True if file UUIDs are to be computed
    :rtype compute_uuids: bool
    :param dbgap_study_id: a valid dbGaP study ID
    :rtype dbgap_study_id: string
    :param recompute_file_extension: True if file extensions are to be recomputed
    :rtype recompute_file_extension: bool
    :param debug: debug flag
    :rtype debug: bool
    """

    __pprint(f"Attempting to process dataset with dataset ID {hubmap_id}")

    if dbgap_study_id:
        print(f"Setting dbGaP study ID to {dbgap_study_id}")

    print(f"Number of cores to be used in this run is {str(ncores)}.")
    pandarallel.initialize(progress_bar=True, nb_workers=ncores)

    metadata = hubmapbags.apis.get_dataset_info(hubmap_id, instance="prod", token=token)
    global directory
    directory = hubmapbags.get_directory(hubmap_id, instance="prod", token=token)
    is_protected = hubmapbags.apis.is_protected(hubmap_id, instance="prod", token=token)
    hubmap_uuid = metadata["uuid"]

    if directory[-1] == "/":
        directory = directory[:-1]

    data_directory = "data"
    print(f"Data directory set to {data_directory}.")
    if not Path(data_directory).exists():
        Path(data_directory).mkdir()

    file = directory.replace("/", "_")
    output_filename = f'{data_directory}/{metadata["uuid"]}.tsv'
    print(f"Saving results to {output_filename}")

    temp_directory = "/local/"
    if not Path(temp_directory).exists():
        temp_directory = ".tmp/"
        if not Path(temp_directory).exists():
            Path(temp_directory).mkdir()
    print(f"Temp directory set to {temp_directory}.")

    if Path(temp_directory + output_filename).exists():
        shutil.copyfile(temp_directory + output_filename, output_filename)
        print(
            f"Found existing temp file {temp_directory + output_filename}. Reusing file."
        )

    if Path(output_filename).exists():
        print(f"Loading dataframe in file {output_filename}.")
        df = pd.read_csv(output_filename, sep="\t", low_memory=False)
        print(f"Processing {str(len(df))} files in directory")
    else:
        print("Creating empty dataframe")
        files = [
            x
            for x in Path(directory).glob("**/*")
            if (x.is_file() or x.is_symlink()) and not x.is_dir()
        ]
        df = pd.DataFrame()
        df["full_path"] = files
        print(f"Populating dataframe with {str(len(df))} files")

    if recompute_file_extension:
        if "extension" in df.keys():
            df = df.drop(["extension"], axis=1)

        if "file_type" in df.keys():
            df = df.drop(["file_type"], axis=1)

        if "data_type" in df.keys():
            df = df.drop(["data_type"], axis=1)

        if "file_format" in df.keys():
            df = df.drop(["file_format"], axis=1)

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get file extensions")

    def __get_relative_path(full_path: str) -> str:
        answer = str(full_path).replace(f"{directory}/", "")
        return answer

    def __get_file_extension(filename: str) -> str:
        extension = Path(filename).suffix

        if extension == ".tiff" or extension == ".tif":
            if str(filename).find("ome.tif") >= 0:
                extension = ".ome.tif"

        if extension == ".gz":
            if str(filename).find(".fastq.gz") >= 0:
                extension = ".fastq.gz"

        return extension

    if "relative_path" not in df.keys():
        df["relative_path"] = df["full_path"].apply(__get_relative_path)

    if "extension" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["extension"] = df["relative_path"].parallel_apply(__get_file_extension)
    else:
        temp = df[df["extension"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["extension"] = temp["full_path"].apply(__get_file_extension)
        else:
            temp["extension"] = temp["full_path"].parallel_apply(__get_file_extension)

        df = __update_dataframe(df, temp, "extension")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get file names")

    def get_filename(filename: str) -> str:
        """
        Get the base filename (without the directory path) from a given filename.

        This method takes a full filename (including the directory path) as input
        and returns only the base filename (i.e., without the directory path).

        :param filename: The full filename including the directory path.
        :type filename: str

        :return: The base filename without the directory path.
        :rtype: str
        """

        return Path(filename).stem + Path(filename).suffix

    if "filename" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["filename"] = df["full_path"].parallel_apply(get_filename)
    else:
        temp = df[df["filename"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["filename"] = temp["full_path"].apply(get_filename)
        else:
            temp["filename"] = temp["full_path"].parallel_apply(get_filename)

        df = __update_dataframe(df, temp, "filename")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Computing file types")

    def __get_file_type(extension: str) -> str:
        """
        Get the type of file based on its extension.

        This method determines the type of file based on its extension.
        It categorizes files as "images", "sequence", or "other".

        :param extension: The file extension (e.g., ".jpg", ".txt").
        :type extension: str

        :return: The type of file ("images", "sequence", or "other").
        :rtype: str
        """

        try:
            images = {
                ".tiff",
                ".png",
                ".tif",
                ".ome.tif",
                ".jpeg",
                ".gif",
                ".ome.tiff",
                "jpg",
                ".jp2",
            }
            if extension in images:
                return "images"
            elif extension.find("fast") > 0:
                return "sequence"
            else:
                return "other"
        except:
            return "other"

    print(f"Processing {str(len(df))} files in directory")
    df["file_type"] = df["extension"].apply(__get_file_type)
    df.to_csv(output_filename, sep="\t", index=False)

    if "file_type" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["file_type"] = df["extension"].parallel_apply(__get_file_type)
    else:
        temp = df[df["file_type"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["file_type"] = temp["extension"].apply(__get_file_type)
        else:
            temp["file_type"] = temp["extension"].parallel_apply(__get_file_type)

        df = __update_dataframe(df, temp, "file_type")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get file creation date")

    def __get_file_creation_date(filename: str) -> str:
        """
        Get the creation date of a file.

        This method retrieves the creation date of a file specified by its 'filename'.
        The creation date is returned as a formatted string representing the date and time.

        :param filename: The full path of the file.
        :type filename: str

        :return: The creation date of the file as a formatted string.
        :rtype: str
        """
        t = os.path.getmtime(str(filename))
        return str(datetime.datetime.fromtimestamp(t))

    if "modification_time" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["modification_time"] = df["full_path"].parallel_apply(
            __get_file_creation_date
        )
    else:
        temp = df[df["modification_time"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["modification_time"] = temp["full_path"].apply(
                __get_file_creation_date
            )
        else:
            temp["modification_time"] = temp["full_path"].parallel_apply(
                __get_file_creation_date
            )

        df = __update_dataframe(df, temp, "modification_time")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get file size")

    def __get_file_size(filename: str) -> int:
        """
        Get the size of a file in bytes.

        This method retrieves the size of a file specified by its 'filename' in bytes.

        :param filename: The full path of the file.
        :type filename: str

        :return: The size of the file in bytes.
        :rtype: int
        """

        return Path(filename).stat().st_size

    if "size" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["size"] = df["full_path"].parallel_apply(__get_file_size)
    else:
        temp = df[df["size"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["size"] = temp["full_path"].apply(__get_file_size)
        else:
            temp["size"] = temp["full_path"].parallel_apply__(get_file_size)

        df = __update_dataframe(df, temp, "size")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get mime-type")

    def __get_mime_type(filename: str) -> str:
        """
        Get the MIME type of a file.

        This method retrieves the MIME type of a file specified by its 'filename'.

        :param filename: The full path of the file.
        :type filename: str

        :return: The MIME type of the file.
        :rtype: str
        """
        mime = magic.Magic(mime=True)
        return mime.from_file(filename)

    if "mime_type" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["mime_type"] = df["full_path"].parallel_apply(__get_mime_type)
    else:
        temp = df[df["mime_type"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["mime_type"] = temp["full_path"].apply(__get_mime_type)
        else:
            temp["mime_type"] = temp["full_path"].parallel_apply(__get_mime_type)

        df = __update_dataframe(df, temp, "mime_type")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint("Get download link for each file")

    def __get_url(filename: str) -> str:
        """
        Get the URL corresponding to a filename.

        This method converts a given 'filename' to its corresponding URL.

        :param filename: The full path of the file.
        :type filename: str

        :return: The URL corresponding to the file.
        :rtype: str
        """

        filename = str(filename)
        return filename.replace(
            "/hive/hubmap/data/public", "https://g-d00e7b.09193a.5898.dn.glob.us"
        )

    if not hubmapbags.apis.is_protected(hubmap_id, instance="prod", token=token):
        if "download_url" not in df.keys():
            print(f"Processing {str(len(df))} files in directory")
            if is_protected:
                df["download_url"] = None
            else:
                df["download_url"] = df["full_path"].parallel_apply(__get_url)
            df.to_csv(output_filename, sep="\t", index=False)
        else:
            temp = df[df["download_url"].isnull()]
            print(f"Processing {str(len(temp))} files of {str(len(df))} files")
            if len(temp) < ncores:
                temp["download_url"] = temp["full_path"].apply(__get_url)
            else:
                temp["download_url"] = temp["full_path"].parallel_apply(__get_url)

            df = __update_dataframe(df, temp, "download_url")

        df.to_csv(output_filename, sep="\t", index=False)
    else:
        print("Dataset is protected. Avoiding computation of download URLs.")
        df["download_url"] = None

    ###############################################################################################################
    import warnings

    warnings.filterwarnings("ignore")

    __pprint("Computing MD5 checksums")

    def __compute_md5sum(filename: str) -> str:
        """
        Compute the MD5 checksum of a file.

        This method computes the MD5 checksum of the contents of the file specified
        by its 'filename'.

        :param filename: The full path of the file.
        :type filename: str

        :return: The MD5 checksum of the file as a hexadecimal string.
        :rtype: str

        :Example:

        >>> file_path = "/path/to/file/example.txt"
        >>> md5_checksum = __compute_md5sum(file_path)
        >>> print(md5_checksum)
        5eb63bbbe01eeed093cb22bb8f5acdc3
        """

        # BUF_SIZE is totally arbitrary, change for your app!
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

        md5 = hashlib.md5()

        if Path(filename).is_file() or Path(filename).is_symlink():
            with open(filename, "rb") as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    md5.update(data)

        return md5.hexdigest()

    def __get_chunk_size(dataframe):
        """
        Get the chunk size based on the length of a DataFrame.

        This method returns a chunk size based on the length of the input 'dataframe'.
        The chunk size is chosen to optimize processing performance.

        :param dataframe: The DataFrame for which the chunk size is calculated.
        :type dataframe: pandas.DataFrame

        :return: The chunk size to be used for processing the DataFrame.
        :rtype: int

        :Example:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        >>> chunk_size = __get_chunk_size(df)
        >>> print(chunk_size)
        10
        """

        if len(dataframe) < 1000:
            return 10
        elif len(dataframe) < 10000:
            return 100
        elif len(dataframe) < 100000:
            return 250
        elif len(dataframe) < 500000:
            return 500
        else:
            return 500

    if len(df) < 100:
        if "md5" in df.keys():
            files = df[df["md5"].isnull()]
        else:
            files = df
        print(f"Number of files to process is {str(len(files))}")

        if len(files) > 0:
            files["md5"] = files["full_path"].parallel_apply(__compute_md5sum)
            df = __update_dataframe(df, files, "md5")
            df.to_csv(output_filename, sep="\t", index=False)
    else:
        if "md5" in df.keys():
            files = df[df["md5"].isnull()]
        else:
            files = df

        if len(files) != 0:
            n = __get_chunk_size(files)
            print(f"Number of files to process is {str(len(files))}")
            if n < 25:
                files["md5"] = files["full_path"].parallel_apply(__compute_md5sum)
                df = __update_dataframe(df, files, "md5")
                df.to_csv(output_filename, sep="\t", index=False)
            else:
                chunks = np.array_split(files, n)
                chunk_counter = 1
                for chunk in chunks:
                    print(
                        f"\nProcessing chunk {str(chunk_counter)} of {str(len(chunks))}"
                    )
                    chunk["md5"] = chunk["full_path"].parallel_apply(__compute_md5sum)
                    df = __update_dataframe(df, chunk, "md5")
                    chunk_counter = chunk_counter + 1

                    if chunk_counter % 10 == 0 or chunk_counter == len(chunks):
                        print("\nSaving chunks to disk")
                        df.to_csv(output_filename, sep="\t", index=False)
        else:
            print("No files left to process")

    ###############################################################################################################
    __pprint("Computing SHA256 checksums")

    def __compute_sha256sum(filename: str) -> str:
        """
        Compute the SHA-256 checksum of a file.

        This method computes the SHA-256 checksum of the contents of the file specified
        by its 'filename'.

        :param filename: The full path of the file.
        :type filename: str

        :return: The SHA-256 checksum of the file as a hexadecimal string.
        :rtype: str

        :Example:

        >>> file_path = "/path/to/file/example.txt"
        >>> sha256_checksum = __compute_sha256sum(file_path)
        >>> print(sha256_checksum)
        a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6
        """

        # BUF_SIZE is totally arbitrary, change for your app!
        BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

        sha256 = hashlib.sha256()
        if Path(filename).is_file() or Path(filename).is_symlink():
            with open(filename, "rb") as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    sha256.update(data)

        return sha256.hexdigest()

    def __get_chunk_size(dataframe):
        """
        Get the chunk size based on the length of a DataFrame.

        This method returns a chunk size based on the length of the input 'dataframe'.
        The chunk size is chosen to optimize processing performance.

        :param dataframe: The DataFrame for which the chunk size is calculated.
        :type dataframe: pandas.DataFrame

        :return: The chunk size to be used for processing the DataFrame.
        :rtype: int

        :Example:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        >>> chunk_size = __get_chunk_size(df)
        >>> print(chunk_size)
        10
        """
        if len(dataframe) < 1000:
            return 10
        elif len(dataframe) < 10000:
            return 100
        elif len(dataframe) < 100000:
            return 250
        elif len(dataframe) < 500000:
            return 500
        else:
            return 500

    if len(df) < 100:
        if "sha256" in df.keys():
            files = df[df["sha256"].isnull()]
        else:
            files = df
        print(f"Number of files to process is {str(len(files))}")

        if len(files) > 0:
            files["sha256"] = files["full_path"].parallel_apply(__compute_sha256sum)
            df = __update_dataframe(df, files, "sha256")
            df.to_csv(output_filename, sep="\t", index=False)
    else:
        if "sha256" in df.keys():
            files = df[df["sha256"].isnull()]
        else:
            files = df

        if not files.empty:
            n = __get_chunk_size(files)
            print(f"Number of files to process is {str(len(files))}")

            if n < 25:
                files["sha256"] = files["full_path"].parallel_apply(__compute_sha256sum)
                df = __update_dataframe(df, files, "sha256")
                df.to_csv(output_filename, sep="\t", index=False)
            else:
                chunks = np.array_split(files, n)

                chunk_counter = 1
                for chunk in chunks:
                    print(
                        f"\nProcessing chunk {str(chunk_counter)} of {str(len(chunks))}"
                    )
                    chunk["sha256"] = chunk["full_path"].parallel_apply(
                        __compute_sha256sum
                    )
                    df = __update_dataframe(df, chunk, "sha256")
                    chunk_counter = chunk_counter + 1

                    if chunk_counter % 10 == 0 or chunk_counter == len(chunks):
                        print("\nSaving chunks to disk")
                        df.to_csv(output_filename, sep="\t", index=False)
        else:
            print("No files left to process")

    ###############################################################################################################
    import requests

    def __generate(
        hubmap_id: str,
        df: pd.DataFrame,
        token: str,
        instance: str = "prod",
        debug: bool = False,
    ) -> pd.DataFrame:
        """
        Generate UUIDs using the uuid-api.

        This method generates UUIDs for files in the input DataFrame 'df' that have
        'file_uuid' values as NaN (null). The generated UUIDs are added to the DataFrame.

        :param hubmap_id: The ID of the HubMap dataset.
        :type hubmap_id: str

        :param df: The DataFrame containing file information with 'file_uuid' column.
        :type df: pd.DataFrame

        :param token: The access token for authentication with the uuid-api.
        :type token: str

        :param instance: The instance to use for generating UUIDs (default is "prod").
        :type instance: str, optional

        :param debug: Enable debug mode for additional information (default is False).
        :type debug: bool, optional

        :return: The DataFrame with generated UUIDs added (or None if nothing to generate).
        :rtype: pd.DataFrame or None

        :Example:

        >>> import pandas as pd
        >>> df = pd.DataFrame({'file_uuid': [None, None, None],
        ...                    'relative_path': ['file1.txt', 'file2.txt', 'file3.txt'],
        ...                    'size': [100, 200, 300],
        ...                    'sha256': ['sha256_1', 'sha256_2', 'sha256_3']})
        >>> hubmap_id = "abc123"
        >>> token = "your_access_token"
        >>> generated_df = __generate(hubmap_id, df, token)
        >>> print(generated_df)
        file_uuid relative_path  size    sha256
        0   uuid_1    file1.txt     100     sha256_1
        1   uuid_2    file2.txt     200     sha256_2
        2   uuid_3    file3.txt     300     sha256_3
        """

        df = df[df["file_uuid"].isnull()]
        if df.empty:
            print("Nothing left to generate.")
            return None

        metadata = hubmapbags.apis.get_dataset_info(
            hubmap_id, instance=instance, token=token, overwrite=False
        )

        URL = "https://uuid.api.hubmapconsortium.org/hmuuid/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
        }

        if len(df) <= 1000:
            if df["file_uuid"].isnull().all():
                file_info = []

                for index, datum in df.iterrows():
                    filename = datum["relative_path"]
                    file_info.append(
                        {
                            "path": filename,
                            "size": datum["size"],
                            "checksum": datum["sha256"],
                            "base_dir": "DATA_UPLOAD",
                        }
                    )

                payload = {}
                payload["parent_ids"] = [metadata["uuid"]]
                payload["entity_type"] = "FILE"
                payload["file_info"] = file_info
                params = {"entity_count": len(file_info)}
                print("Generating UUIDs")
                r = requests.post(
                    URL,
                    params=params,
                    headers=headers,
                    data=json.dumps(payload),
                    allow_redirects=True,
                    timeout=120,
                )
                j = json.loads(r.text)
        else:
            print(
                "Data frame has "
                + str(len(df))
                + " items. Partitioning into smaller chunks."
            )

            n = 100  # chunk row size
            dfs = [df[i : i + n] for i in range(0, df.shape[0], n)]

            print("Generating UUIDs")
            counter = 0
            for frame in dfs:
                counter = counter + 1
                print(
                    "Generating UUIDs for partition "
                    + str(counter)
                    + " of "
                    + str(len(dfs))
                    + "."
                )

                file_info = []
                for index, datum in frame.iterrows():
                    filename = datum["relative_path"]
                    file_info.append(
                        {
                            "path": filename,
                            "size": datum["size"],
                            "checksum": datum["sha256"],
                            "base_dir": "DATA_UPLOAD",
                        }
                    )

                payload = {}
                payload["parent_ids"] = [metadata["uuid"]]
                payload["entity_type"] = "FILE"
                payload["file_info"] = file_info
                params = {"entity_count": len(file_info)}

                if frame["file_uuid"].isnull().all():
                    r = requests.post(
                        URL,
                        params=params,
                        headers=headers,
                        data=json.dumps(payload),
                        allow_redirects=True,
                        timeout=120,
                    )
                    j = json.loads(r.text)
                else:
                    print("HuBMAP UUIDs chunk is populated. Skipping recomputation.")

        return df

    def __populate_local_file_with_remote_uuids(
        df: pd.DataFrame, uuids: dict
    ) -> pd.DataFrame:
        """
        Helper function that populates (but does not generate) a local pickle file with remote UUIDs.
        """

        uuids = pd.DataFrame.from_dict(uuids)

        if uuids.empty:
            print("There are no UUIDs for this dataset in UUID service")
            df["file_uuid"] = None
        else:
            uuids = uuids[uuids["base_dir"] == "DATA_UPLOAD"]
            uuids = uuids[["file_uuid", "path"]]
            uuids.rename(columns={"path": "relative_path"}, inplace=True)
            df = df.merge(uuids, on="relative_path", how="left")
            df.rename(columns={"file_uuid_y": "file_uuid"}, inplace=True)
            df = df[
                [
                    "file_uuid",
                    "full_path",
                    "relative_path",
                    "filename",
                    "extension",
                    "file_type",
                    "size",
                    "mime_type",
                    "modification_time",
                    "md5",
                    "sha256",
                    "download_url",
                ]
            ]
        return df

    provenance = hubmapbags.apis.get_provenance_info(
        hubmap_id, instance="prod", token=token
    )
    if compute_uuids:
        __pprint("Generating or pulling UUIDs from HuBMAP UUID service")
        if provenance["dataset_data_types"][0].find("[Salmon]") >= 0:
            print(
                "This derived dataset is the result from running Salmon. Avoiding computation of zarr files."
            )
        elif provenance["dataset_data_types"][0].find("[Cytokit + SPRM]") >= 0:
            print(
                "This derived dataset is the result from running Cytokit + SPRM. Avoiding computation of zarr files."
            )
        else:
            if not "file_uuid" in df.keys():
                df["file_uuid"] = None

            if "file_uuid" in df.keys() and len(df[df["file_uuid"].isnull()]) > 0:
                print(
                    "There are missing UUIDs from this data frame. Generating or pulling UUIDs."
                )
                uuids = hubmapbags.uuids.get_uuids(
                    hubmap_id, instance="prod", token=token
                )
                df = __populate_local_file_with_remote_uuids(df, uuids)

                if not df[df["file_uuid"].isnull()].empty:
                    __generate(hubmap_id, df, instance="prod", token=token, debug=True)
                    df = __populate_local_file_with_remote_uuids(df, uuids)
            else:
                print(
                    "Dataframe is populated with UUIDs. Avoiding generation or retrieval."
                )

        df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    if dbgap_study_id:
        __pprint(f"Populating dbGap study ID {dbgap_study_id}")
        print("Populating dataframe")
        df.loc[df["extension"] == ".fastq.gz", "dbgap_study_id"] = dbgap_study_id
    else:
        df["dbgap_study_id"] = None

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint(f"Populating file format with EDAM ontology")

    def __get_file_format(extension: str) -> str:
        fileformats = {
            ".ome.tiff": "http://edamontology.org/format_3727",
            ".ome.tif": "http://edamontology.org/format_3727",
            ".tif": "http://edamontology.org/format_3591",
            ".tiff": "http://edamontology.org/format_3591",
            ".hdf5": "http://edamontology.org/format_3590",
            ".hdf": "http://edamontology.org/format_3873",
            ".jpg": "http://edamontology.org/format_3579",
            ".png": "http://edamontology.org/format_3873",
            ".tar": "http://edamontology.org/format_3981",
            ".tgz": "http://edamontology.org/format_3989",
            ".gz": "http://edamontology.org/format_3989",
            ".fastq.gz": "http://edamontology.org/format_3989",
            ".pdf": "http://edamontology.org/format_3508",
            ".ibd": "http://edamontology.org/format_3839",
            ".gif": "http://edamontology.org/format_3467",
            ".zip": "http://edamontology.org/format_3987",
            ".xls": "http://edamontology.org/format_3468",
            ".xlsx": "http://edamontology.org/format_3620",
            ".tex": "http://edamontology.org/format_3817",
            ".json": "http://edamontology.org/format_3464",
            ".yaml": "http://edamontology.org/format_3750",
            ".yml": "http://edamontology.org/format_3750",
            ".log": "http://edamontology.org/data_3671",
            ".txt": "http://edamontology.org/data_3671",
            ".csv": "http://edamontology.org/format_3752",
            ".tsv": "http://edamontology.org/format_3475",
            ".mtx": "http://edamontology.org/format_3916",
            ".xml": "http://edamontology.org/format_2332",
        }

        if extension in fileformats.keys():
            return fileformats[extension]
        else:
            return None

    if "file_format" not in df.keys():
        print(f"Processing {str(len(df))} files in directory")
        df["file_format"] = df["extension"].parallel_apply(__get_file_format)
    else:
        temp = df[df["file_format"].isnull()]
        print(f"Processing {str(len(temp))} files of {str(len(df))} files")
        if len(temp) < ncores:
            temp["file_format"] = temp["full_path"].apply(__get_file_format)
        else:
            temp["file_format"] = temp["full_path"].parallel_apply(__get_file_format)

        df = __update_dataframe(df, temp, "file_format")

    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint(f"Populating data format with EDAM ontology")
    print("This has not been implemented yet.")
    df["data_format"] = None
    df.to_csv(output_filename, sep="\t", index=False)

    ###############################################################################################################
    __pprint(f"Populating HuBMAP ID and UUID")
    df["dataset_id"] = hubmap_id
    df["dataset_uuid"] = hubmap_uuid
    df.to_csv(output_filename, sep="\t", index=False)
    print("Done populating dataframe.")
    df = df.drop(["dataset_id", "dataset_uuid"], axis=1)

    ###############################################################################################################
    __pprint("Computing dataset level statistics")
    import humanize

    def get_url(filename: str) -> str:
        return filename.replace("/bil/data/", "https://download.brainimagelibrary.org/")

    def __get_dataset_type(hubmap_id: str, token: str, instance: str = "prod"):
        metadata = hubmapbags.apis.get_dataset_info(
            hubmap_id, instance="prod", token=token
        )

        if metadata["direct_ancestors"][0]["entity_type"] == "Sample":
            return "Primary"
        elif metadata["direct_ancestors"][0]["entity_type"] == "Dataset":
            return "Derived"
        else:
            return "Unknown"

    def generate_dataset_uuid(directory: str) -> str:
        if directory[-1] == "/":
            directory = directory[:-1]

        return str(uuid.uuid5(uuid.NAMESPACE_DNS, directory))

    dataset = {}
    dataset["hubmap_id"] = hubmap_id
    dataset["uuid"] = metadata["uuid"]
    dataset["status"] = metadata["status"]
    dataset["dataset_type"] = __get_dataset_type(
        hubmap_id, instance="prod", token=token
    )
    dataset["is_protected"] = is_protected
    dataset["directory"] = directory

    if "doi_url" in metadata.keys():
        dataset["doi_url"] = metadata["doi_url"]

    dataset["number_of_files"] = len(df)
    dataset["size"] = df["size"].sum()
    dataset["pretty_size"] = humanize.naturalsize(dataset["size"], gnu=True)
    dataset["frequencies"] = df["extension"].value_counts().to_dict()

    provenance = hubmapbags.apis.get_provenance_info(
        hubmap_id, instance="prod", token=token
    )
    dataset["data_type"] = provenance["dataset_data_types"][0]
    dataset["creation_date"] = provenance["dataset_date_time_created"]
    dataset["group_name"] = provenance["dataset_group_name"]

    df["full_path"] = df["full_path"].astype(str)
    files = df.to_dict("records")
    dataset["manifest"] = files

    output_filename = f'{data_directory}/{metadata["uuid"]}.json'
    print(f"Saving results to {output_filename}")
    with open(output_filename, "w") as ofile:
        json.dump(
            dataset,
            ofile,
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            cls=NumpyEncoder,
        )

    # compressed
    with gzip.open(f"{output_filename}.gz", "wt") as f:
        f.write(str(dataset))

    backup_destination = "/hive/hubmap/bdbags/inventory"
    if backup and Path(backup_destination).exists():
        print(f"Backing up to {backup_destination}")
        output_filename = f'{backup_destination}/{metadata["uuid"]}.tsv'
        df.to_csv(output_filename, sep="\t", index=False)

        output_filename = f'{backup_destination}/{metadata["uuid"]}.json'
        print(f"Saving results to {output_filename}")
        with open(output_filename, "w") as ofile:
            json.dump(
                dataset,
                ofile,
                indent=4,
                sort_keys=True,
                ensure_ascii=False,
                cls=NumpyEncoder,
            )

        # compressed
        with gzip.open(f"{output_filename}.gz", "wt") as f:
            f.write(str(dataset))

    print("\nDone\n")

    return df
