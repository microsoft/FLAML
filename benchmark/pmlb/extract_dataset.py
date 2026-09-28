import gzip
import os

import pandas as pd

basepath = "benchmark/pmlb/datasets/pmlb/datasets"
tsv_path = os.path.join("benchmark", "pmlb", "csv_datasets")
os.makedirs(tsv_path, exist_ok=True)
for dataset_folder in os.listdir(basepath):
    fp = os.path.join(basepath, dataset_folder, dataset_folder + ".tsv.gz")
    f = gzip.GzipFile(fp)
    open(os.path.join(tsv_path, dataset_folder + ".tsv"), "wb").write(f.read())
for tsv_file in os.listdir(tsv_path):
    pd.read_csv(os.path.join(tsv_path, tsv_file), sep="\t").to_csv(
        os.path.join(tsv_path, tsv_file.replace("tsv", "csv")), index=False
    )
    os.remove(os.path.join(tsv_path, tsv_file))
