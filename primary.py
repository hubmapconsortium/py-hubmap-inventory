import hubmapinventory
import hubmapbags

token = ""
ncores = 25

assay_types = hubmapbags.apis.get_assay_types(token=token)
assay_types.remove("CODEX")
assay_types.remove("codex_cytokit")
assay_types.remove("codex_cytokit_v1")

compute_uuids = True
for assay_type in assay_types:
    hubmapbags.utilities.pprint(assay_type)
    print("Retrieving dataset IDs. This might take a while. Be patient.")
    datasets = hubmapbags.apis.get_hubmap_ids(assay_type, token=token)

    for dataset in datasets:
        try:
            if (
                dataset["status"] == "Published"
                and not dataset["is_protected"]
                and dataset["is_primary"]
            ):
                df = hubmapinventory.inventory.create(
                    dataset["hubmap_id"],
                    token=token,
                    ncores=ncores,
                    compute_uuids=compute_uuids,
                    recompute_file_extension=True,
                )
            else:
                print(f'Avoiding computation of dataset {dataset["hubmap_id"]}.')
        except:
            print(f'Failed to process dataset {dataset["hubmap_id"]}.')
