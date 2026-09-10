"""
fetch_nhanes.py
================
Downloads the NHANES 2017-2018 component files needed by
`src.datasets.load_nhanes`, merges them on the respondent ID (SEQN), and
writes a single small CSV (`data/nhanes_2017_2018.csv`).

NHANES files are released per 2-year "cycle" as separate SAS XPT files (one
per questionnaire/exam module), each keyed by SEQN. The "_J" suffix denotes
the 2017-2018 cycle.

Files used:
  DEMO_J  Demographics       (sex, age, race/ethnicity, education, income)
  DIQ_J   Diabetes questionnaire (outcome)
  BPQ_J   Blood pressure & cholesterol questionnaire
  BMX_J   Body measures (BMI)
  SMQ_J   Smoking questionnaire
  PAQ_J   Physical activity questionnaire
  HIQ_J   Health insurance questionnaire

Run from the repo root:
    python scripts/fetch_nhanes.py
"""

import os
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"
FILES = ["DEMO_J", "DIQ_J", "BPQ_J", "BMX_J", "SMQ_J", "PAQ_J", "HIQ_J"]

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = os.path.join(DATA_DIR, "nhanes_raw")
OUT_PATH = os.path.join(DATA_DIR, "nhanes_2017_2018.csv")

# Columns kept from each component file (SEQN is the merge key in all of them).
COLUMNS = {
    "DEMO_J": ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "DMDEDUC2", "INDFMPIR"],
    "DIQ_J": ["SEQN", "DIQ010"],
    "BPQ_J": ["SEQN", "BPQ020", "BPQ080"],
    "BMX_J": ["SEQN", "BMXBMI"],
    "SMQ_J": ["SEQN", "SMQ020"],
    "PAQ_J": ["SEQN", "PAQ605"],
    "HIQ_J": ["SEQN", "HIQ011"],
}


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    frames = {}
    for name in FILES:
        out_path = os.path.join(RAW_DIR, f"{name}.xpt")
        if not os.path.exists(out_path):
            print(f"Downloading {name}.xpt ...")
            urllib.request.urlretrieve(f"{BASE_URL}{name}.xpt", out_path)
        frames[name] = pd.read_sas(out_path, format="xport")[COLUMNS[name]]

    df = frames["DEMO_J"]
    for name in FILES[1:]:
        df = df.merge(frames[name], on="SEQN", how="left")

    print(f"Writing {OUT_PATH} ({df.shape[0]} rows x {df.shape[1]} cols)")
    df.to_csv(OUT_PATH, index=False)

    print("Cleaning up raw downloads ...")
    shutil.rmtree(RAW_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
