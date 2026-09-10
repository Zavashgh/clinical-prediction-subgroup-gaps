"""
fetch_cchs.py
=============
Builds `data/cchs_2019_2020_subset.csv` from the CCHS 2019-2020 PUMF.

Unlike the other `fetch_*.py` scripts, the CCHS PUMF cannot be downloaded
automatically -- Statistics Canada distributes it through Borealis Data
(https://borealisdata.ca), which requires a free account and a one-click
acceptance of the Statistics Canada Open Licence.

Manual steps (one-time):
  1. Create a free account at https://borealisdata.ca
  2. Go to the dataset page:
     https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/ZVCGBK
     ("Canadian Community Health Survey, 2019-2020: Annual Component")
  3. Click "Access Dataset" -> "Original Format ZIP" (~1.4 GB), accept the
     Open Licence, and download.
  4. Extract the ZIP. The SPSS data file is at:
       Data/spss/cchs_201920_pumf.sav

Then run this script, passing the path to that .sav file:
    python scripts/fetch_cchs.py "C:\\path\\to\\cchs_201920_pumf.sav"

This extracts only the columns `src.datasets.load_cchs` needs (691 columns
-> 12) and writes the much smaller `data/cchs_2019_2020_subset.csv`.
Requires `pyreadstat` (pip install pyreadstat).
"""

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_PATH = os.path.join(DATA_DIR, "cchs_2019_2020_subset.csv")

# Columns used by src.datasets.load_cchs
COLUMNS_NEEDED = [
    "DHH_SEX",    # sex at birth (protected attribute)
    "DHHGAGE",    # age group
    "CCC_095",    # has diabetes (outcome)
    "CCC_065",    # has high blood pressure
    "CCC_075",    # has high blood cholesterol
    "HWTDGBCC",   # BMI classification (under/normal vs overweight/obese)
    "EHG2DVH3",   # education (household, 3 levels)
    "INCDGHH",    # household income, grouped
    "SMKDVSTY",   # smoking status
    "PAADVMVA",   # moderate-to-vigorous physical activity minutes / week
    "PHC_020",    # has a regular health care provider
    "GEOGPRV",    # province
]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    sav_path = sys.argv[1]
    print(f"Reading {sav_path} ...")
    df = pd.read_spss(sav_path, usecols=COLUMNS_NEEDED)

    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Writing {OUT_PATH} ({df.shape[0]} rows x {df.shape[1]} cols)")
    df.to_csv(OUT_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
