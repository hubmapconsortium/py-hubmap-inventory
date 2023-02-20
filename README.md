# hubmap-inventory

## Examples

### `inventory-builder.py`
To build an inventory file from a public CODEX dataset you need the dataset HuBMAP ID and a token

```
export TOKEN=<this-is-my-token>
python ./manifest-builder.py -t $TOKEN -n 25 -i HBM342.FSLD.938
```

See the data folder for an example of both the TSV and JSON files produced by this script.

### `histogram.sh`
Uses `lowcharts` to generate histograms on terminal

```
bash ./histogram.sh mime-type data/f92a933da30d4bb3a4f8d031f2bc5d3d.tsv
Each ∎ represents a count of 1

[application/xml] [14] ∎∎∎∎∎∎∎∎∎∎∎∎∎∎
[     text/plain] [ 5] ∎∎∎∎∎
[      mime-type] [ 1] ∎

bash ./histogram.sh extension data/f92a933da30d4bb3a4f8d031f2bc5d3d.tsv
Each ∎ represents a count of 1

[    .mzML] [14] ∎∎∎∎∎∎∎∎∎∎∎∎∎∎
[     .csv] [ 2] ∎∎
[     .tsv] [ 2] ∎∎
[extension] [ 1] ∎
[    .orig] [ 1] ∎

bash ./histogram.sh filetype data/f92a933da30d4bb3a4f8d031f2bc5d3d.tsv
Each ∎ represents a count of 1

[   other] [19] ∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎∎
[filetype] [ 1] ∎
```
