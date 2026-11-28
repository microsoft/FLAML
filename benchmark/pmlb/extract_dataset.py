"""Extract PMLB TSV archives without removing existing CSV outputs."""

import os
import tempfile
from pathlib import Path

import pandas as pd


def extract_datasets(basepath="benchmark/pmlb/datasets/pmlb/datasets", tsv_path="benchmark/pmlb/csv_datasets"):
    destination = Path(tsv_path)
    destination.mkdir(parents=True, exist_ok=True)
    for dataset_folder in Path(basepath).iterdir():
        if not dataset_folder.is_dir():
            continue
        source = dataset_folder / f"{dataset_folder.name}.tsv.gz"
        output = destination / f"{dataset_folder.name}.csv"
        if output.exists():
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=".pmlb-", suffix=".csv", dir=destination)
        os.close(descriptor)
        try:
            pd.read_csv(source, sep="\t", compression="gzip").to_csv(temporary, index=False)
            try:
                os.link(temporary, output)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)


if __name__ == "__main__":
    extract_datasets()
