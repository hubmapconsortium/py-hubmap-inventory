import sys
import hubmapbags
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a','--assay_type',dest='assay_type',help='Assay type')
parser.add_argument('-t', '--token',dest='token', help='Token')
args = parser.parse_args()

assay_type = args.assay_type
token = args.token

ids = hubmapbags.apis.get_hubmap_ids(assay_type, token=token)
for id in ids:
	if id['status'] == 'Published' and not hubmapbags.apis.is_protected( id['hubmap_id'], instance='prod', token=token):
		file=f'{id["uuid"]}.json'
		print(f'Removing {id["uuid"]}.json')
		if Path(file).exists():
			Path(file).unlink()
		file=f'{id["uuid"]}.tsv'
		print(f'Removing {id["uuid"]}.tsv')
		if Path(file).exists():
			Path(file).unlink()
