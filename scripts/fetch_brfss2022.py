"""
fetch_brfss2022.py
===================
Reproduces `data/brfss2022_subset.csv` from the public BRFSS 2022 annual
data file.

The full annual file (`LLCP2022.XPT`, ~1.16 GB, 445,132 respondents x 328
columns) is too large to keep in the repo, so this script downloads it,
extracts only the columns `src.datasets.load_brfss` needs, writes the
much smaller CSV subset, and deletes the raw download.

Run from the repo root:
    python scripts/fetch_brfss2022.py
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ZIP_PATH = os.path.join(DATA_DIR, "brfss2022.zip")
RAW_DIR = os.path.join(DATA_DIR, "brfss2022_raw")
OUT_PATH = os.path.join(DATA_DIR, "brfss2022_subset.csv")

# Columns used by src.datasets.load_brfss
COLUMNS_NEEDED = [
    "SEXVAR", "_MICHD", "DIABETE4", "_AGEG5YR", "_RACEGR4", "_INCOMG1", "_EDUCAG",
    "CVDSTRK3", "ASTHMA3", "CHCCOPD3", "CHCKDNY2", "HAVARTH4", "ADDEPEV3",
    "_BMI5", "GENHLTH", "PHYSHLTH", "MENTHLTH", "DIFFWALK",
    "_TOTINDA", "_RFDRHV8", "SMOKE100", "MEDCOST1", "_HLTHPLN", "_HCVU652",
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

    print("Extracting ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)

    xpt_path = os.path.join(RAW_DIR, "LLCP2022.XPT")

    print("Reading XPT in chunks and selecting columns ...")
    chunks = []
    for chunk in pd.read_sas(xpt_path, format="xport", chunksize=50000):
        avail = [c for c in COLUMNS_NEEDED if c in chunk.columns]
        chunks.append(chunk[avail].copy())
    df = pd.concat(chunks, ignore_index=True)

    print(f"Writing {OUT_PATH} ({df.shape[0]} rows x {df.shape[1]} cols)")
    df.to_csv(OUT_PATH, index=False)

    print("Cleaning up raw download ...")
    os.remove(ZIP_PATH)
    shutil.rmtree(RAW_DIR)

    print("Done.")


if __name__ == "__main__":
    main()
