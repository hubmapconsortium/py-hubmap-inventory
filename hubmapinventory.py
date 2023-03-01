import os.path
import glob
import subprocess
from PIL import Image
import fnmatch
import tabulate
import pathlib
import numpy as np
import os
import pickle
import pandas as pd
import hashlib
import math
import hubmapbags
import datetime
import time
import uuid
import shutil
from datetime import date
import warnings
from pathlib import Path
from warnings import resetwarnings, warn as warning
from pandarallel import pandarallel
#from joblib import Parallel, delayed
from tqdm import tqdm
import json
import argparse
import sys

###############################################################################################################
def __pprint(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' *row] + ['+'])
    result= '\n' + h + '\n'"|"+msg+"|"'\n' + h
    print(result)

def __update_dataframe(dataset, temp, key):
	for index, datum in temp.iterrows():
		dataset.loc[index,key] = temp.loc[index,key]
	return dataset

###############################################################################################################
def get_relative_path( fullpath ):
	directory2 = directory
	if directory2[-1] != '/':
		directory2 += '/'

	try:
		answer =fullpath.replace( directory2, '' )
		return answer
	except Exception as e:
		print(e)
		print(fullpath)
		return ''

def __get_file_extension(filename):
	extension = None
	if Path(filename).is_file() or Path(filename).is_symlink():
		extension = Path(filename).suffix

		if extension == '.tiff' or extension == '.tif':
			if str(filename).find('ome.tif') >= 0:
				extension = '.ome.tif'

		if str(filename).find('fastq.gz') >= 0:
			extension = 'fastq.gz'

	return extension

def get_file_extensions(df):
    if 'extension' not in df.keys():
        print(f'Processing {str(len(df))} files in directory')
        df['extension'] = df['relativepath'].parallel_apply(__get_file_extension)
    else:
        temp = df[df['extension'].isnull()]
        print(f'Processing {str(len(temp))} files of {str(len(df))} files')
        if len(temp) < ncores:
            temp['extension'] = temp['fullpath'].apply(__et_file_extension)
        else:
            temp['extension'] = temp['fullpath'].parallel_apply(__get_file_extension)

        df = __update_dataframe(df, temp)

    return df

###############################################################################################################
def create( hubmap_id, token=None, ncores=2, compute_uuids=False, dbgap_study_id=None, debug=False ):
    __pprint(f'Attempting to process dataset with dataset ID {hubmap_id}')

    if dbgap_study_id:
        print(f'Setting dbGaP study ID to {dbgap_study_id}')

    print(f'Number of cores to be used in this run is {str(ncores)}.')

    metadata = hubmapbags.apis.get_dataset_info( hubmap_id, instance='prod', token=token )
    directory = hubmapbags.get_directory( hubmap_id, instance='prod', token=token )
    is_protected = hubmapbags.apis.is_protected( hubmap_id, instance='prod', token=token )

    if directory[-1] == '/':
        directory = directory[:-1]

    data_directory = '.data'
    print(f'Data directory set to {data_directory}.')
    if not Path(data_directory).exists():
        Path(data_directory).mkdir()

    file = directory.replace('/', '_')
    output_filename = metadata['uuid'] + '.tsv'

    temp_directory = '/local/'
    if not Path(temp_directory).exists():
        temp_directory = '.tmp/'
        if not Path(temp_directory).exists():
            Path(temp_directory).mkdir()
    print(f'Temp directory set to {temp_directory}.')

    if Path(temp_directory + output_filename).exists():
        shutil.copyfile(temp_directory + output_filename, output_filename)
        print(f'Found existing temp file {temp_directory + output_filename}. Reusing file.')

    if Path(output_filename).exists():
        print(f'Loading dataframe in file {output_filename}.')
        df = pd.read_csv(output_filename, sep='\t', low_memory=False)
        print(f'Processing {str(len(df))} files in directory')
    else:
        print('Creating empty dataframe')
        files = [x for x in Path(directory).glob('**/*') if (x.is_file() or x.is_symlink()) and not x.is_dir()]
        df = pd.DataFrame()
        df['fullpath']= files
        print(f'Populating dataframe with {str(len(df))} files.')

    df = get_file_extensions(df)
    df.to_csv( output_filename, sep='\t', index=False )