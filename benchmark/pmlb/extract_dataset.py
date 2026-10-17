"""Extract PMLB TSV archives without removing existing CSV outputs."""

import gzip
import shutil
from pathlib import Path

import pandas as pd


def extract_datasets(basepath="benchmark/pmlb/datasets/pmlb/datasets", tsv_path="benchmark/pmlb/csv_datasets"):
    destination = Path(tsv_path)
    destination.mkdir(parents=True, exist_ok=True)
    for dataset_folder in Path(basepath).iterdir():
        if not dataset_folder.is_dir():
            continue
        source = dataset_folder / f"{dataset_folder.name}.tsv.gz"
        with gzip.open(source, "rb") as compressed, (destination / f"{dataset_folder.name}.tsv").open("wb") as output:
            shutil.copyfileobj(compressed, output)
    for tsv_file in destination.glob("*.tsv"):
        pd.read_csv(tsv_file, sep="\t").to_csv(tsv_file.with_suffix(".csv"), index=False)
        tsv_file.unlink()


if __name__ == "__main__":
    extract_datasets()
