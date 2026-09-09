"""
datasets.py
===========
Dataset-specific loaders. Each loader takes a path to a local CSV (never
committed -- see .gitignore) and returns a dict describing how to feed the
dataset into `pipeline.run_fairness_analysis`:

  df               : cleaned pandas DataFrame, one row per patient
  feature_cols     : columns used as model features
  target_col       : name of the binary outcome column (1 = event present)
  group_col        : protected-attribute column used for the fairness split
  group_labels     : {raw value: human-readable label}, e.g. {1: "Male", 0: "Female"}
  covariate_blocks : {block name: [columns]} used by the case-mix waterfall
  name             : human-readable dataset name (for titles/tables)

Keeping all dataset-specific cleaning here means `pipeline.py` stays
generic and every dataset goes through the exact same analysis code.
"""

from pathlib import Path

import numpy as np
import pandas as pd


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_data_path(path, filename):
    """Resolve loader paths without depending on the caller's working directory.

    Both ``../data/<filename>`` and ``data/<filename>`` spellings resolve to
    the same project-local file for backward compatibility. Other relative
    paths are resolved from the project root; absolute paths remain supported
    for deliberately external inputs.
    """
    if path is None:
        return _PROJECT_ROOT / "data" / filename
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if len(candidate.parts) >= 2 and candidate.parts[-2:] == ("data", filename):
        return _PROJECT_ROOT / "data" / filename
    return (_PROJECT_ROOT / candidate).resolve()


def load_cdc_diabetes(path=None):
    """
    CDC Diabetes Health Indicators (BRFSS 2015), 253,680 rows.

    The file as downloaded ships the 3-class outcome `Diabetes_012`
    (0 = no diabetes, 1 = prediabetes, 2 = diabetes), NOT the binary
    `Diabetes_binary` column some dataset variants use. We binarize using
    the standard convention for this dataset: prediabetes and diabetes are
    both coded as the positive class.

    Protected attribute: Sex (1 = male, 0 = female), as coded in BRFSS.
    """
    df = pd.read_csv(_resolve_data_path(path, "cdc_diabetes.csv"))

    if "Diabetes_binary" not in df.columns:
        if "Diabetes_012" not in df.columns:
            raise KeyError(
                f"Expected 'Diabetes_binary' or 'Diabetes_012' in columns, got: {list(df.columns)}"
            )
        df["Diabetes_binary"] = (df["Diabetes_012"] >= 1).astype(int)
        df = df.drop(columns=["Diabetes_012"])

    target_col = "Diabetes_binary"
    group_col = "Sex"
    group_labels = {1: "Male", 0: "Female"}

    # Exclude the target and the protected attribute from model features.
    # Sex is excluded from the feature set (a "blinded" model) so that any
    # subgroup gap reflects differences in outcome/case-mix across groups
    # rather than the model directly conditioning on sex.
    feature_cols = [c for c in df.columns if c not in (target_col, group_col)]

    # Used by the case-mix waterfall (adjustments.case_mix_waterfall).
    # Blocks are added cumulatively in this order.
    covariate_blocks = {
        "demographics": ["Age", "Education", "Income"],
        "comorbidities": ["HighBP", "HighChol", "BMI", "Stroke", "HeartDiseaseorAttack",
                          "GenHlth", "PhysHlth", "MentHlth", "DiffWalk"],
        "behavioral": ["Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump"],
        "access": ["AnyHealthcare", "NoDocbcCost", "CholCheck"],
    }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "group_labels": group_labels,
        "covariate_blocks": covariate_blocks,
        "cluster_col": None,
        # This file is fully numeric with no missing values and no categorical
        # source columns, so the training-fitted preprocessor has nothing to
        # learn here. The spec is declared for uniformity across loaders.
        "preprocess_spec": {"categorical": []},
        "name": "CDC Diabetes Health Indicators (BRFSS 2015)",
    }


# 23 diabetes-medication columns in the Diabetes 130-US Hospitals dataset.
# Each is recorded as "No" / "Down" / "Steady" / "Up" (dose direction during
# the encounter). We encode this as an ordinal "dose intensity" 0-3 -- a
# simplification (the categories are not strictly ordered in a clinical
# sense), but it keeps the feature space small and is documented here so the
# choice is easy to revisit.
_DIABETES130_DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

_DOSE_MAP = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}
_AGE_MIDPOINT = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35, "[40-50)": 45,
    "[50-60)": 55, "[60-70)": 65, "[70-80)": 75, "[80-90)": 85, "[90-100)": 95,
}


def load_diabetes130(path=None):
    """
    Diabetes 130-US Hospitals (UCI #296), 1999-2008, 101,766 encounters.

    Outcome: 30-day readmission. `readmitted` is "<30" / ">30" / "NO" in the
    raw data; we binarize as the standard convention for this dataset:
    `readmit_30d = 1` if `readmitted == "<30"`, else 0 (">30" and "NO" are
    both treated as "not an early readmission").

    Protected attribute: `gender` ("Male" / "Female"; the 3 rows coded
    "Unknown/Invalid" are dropped). `race` is also retained as a one-hot
    feature block for the case-mix waterfall, with missing values ("?")
    coded as their own "Unknown" category.

    Feature engineering (kept simple/explicit for reproducibility):
      - `age` (10-year bins) -> bin midpoint, a single numeric feature.
      - The 23 diabetes-medication columns -> ordinal "dose intensity" 0-3
        (No/Down/Steady/Up).
      - `max_glu_serum`, `A1Cresult` -> ordinal 0-3 (None/Norm/elevated/high).
      - `change`, `diabetesMed` -> binary 0/1.
      - `race` -> retained as a raw categorical source column; the
        one-hot expansion is fitted on training rows only after the split.
      - `patient_nbr` is RETAINED as a grouping identifier only. It is never
        a predictor. It exists so the split, selection CV folds, calibration
        folds, and bootstrap can respect repeated encounters per patient.
      - High-cardinality / high-missingness columns dropped:
        `encounter_id` (identifier), `weight` (97% missing),
        `payer_code` (40% missing), `medical_specialty` (49% missing),
        `diag_1`, `diag_2`, `diag_3` (high-cardinality ICD9 codes -- not
        used in this MVP; `number_diagnoses` is kept as a comorbidity-count
        summary).
    """
    df = pd.read_csv(_resolve_data_path(path, "diabetes_130_hospitals.csv"))

    # Drop the 3 rows with unusable gender values.
    df = df[df["gender"] != "Unknown/Invalid"].copy()

    # --- Outcome -------------------------------------------------------
    target_col = "readmit_30d"
    df[target_col] = (df["readmitted"] == "<30").astype(int)

    # --- Protected attribute --------------------------------------------
    group_col = "gender"
    group_labels = {"Male": "Male", "Female": "Female"}

    # --- Feature engineering ---------------------------------------------
    df["age_numeric"] = df["age"].map(_AGE_MIDPOINT)

    for col in _DIABETES130_DRUG_COLS:
        df[col] = df[col].map(_DOSE_MAP)

    df["max_glu_serum"] = df["max_glu_serum"].fillna("None").map({"None": 0, "Norm": 1, ">200": 2, ">300": 3})
    df["A1Cresult"] = df["A1Cresult"].fillna("None").map({"None": 0, "Norm": 1, ">7": 2, ">8": 3})

    df["change"] = df["change"].map({"No": 0, "Ch": 1})
    df["diabetesMed"] = df["diabetesMed"].map({"No": 0, "Yes": 1})

    # Fixed recoding only: "?" is a literal sentinel in this file, not a
    # learned statistic. The one-hot expansion itself is deferred to the
    # training-fitted preprocessor (src/preprocessing.py).
    df["race"] = df["race"].replace("?", "Unknown")

    utilization_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
        "number_outpatient", "number_emergency", "number_inpatient",
    ]
    administrative_cols = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    medication_cols = _DIABETES130_DRUG_COLS + ["max_glu_serum", "A1Cresult", "change", "diabetesMed"]

    # "race" is an expansion point: the training-fitted preprocessor inserts
    # its learned indicator columns at exactly this position, reproducing the
    # column ordering the original loader produced.
    feature_cols = (
        ["age_numeric", "number_diagnoses", "race"]
        + utilization_cols
        + administrative_cols
        + medication_cols
    )

    # `patient_nbr` is RETAINED as a grouping identifier only. It is absent
    # from `feature_cols` and from every covariate block, so it can never
    # enter the design matrix; it exists solely so the split, the selection
    # CV folds, the calibration folds, and the bootstrap can respect the fact
    # that one patient contributes several encounters.
    cluster_col = "patient_nbr"
    drop_cols = [
        "encounter_id", "weight", "payer_code", "medical_specialty",
        "diag_1", "diag_2", "diag_3", "readmitted", "age",
    ]
    df = df.drop(columns=drop_cols)

    covariate_blocks = {
        "demographics": ["age_numeric", "race"],
        "comorbidities": ["number_diagnoses"],
        "utilization": utilization_cols,
        "medications": medication_cols,
        "administrative": administrative_cols,
    }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "group_labels": group_labels,
        "covariate_blocks": covariate_blocks,
        "cluster_col": cluster_col,
        "preprocess_spec": {
            "categorical": [{"source": "race", "prefix": "race", "drop_first": True}],
        },
        "name": "Diabetes 130-US Hospitals (1999-2008)",
    }


# BRFSS 2022 uses the standard "1 = Yes, 2 = No, 7 = Don't know, 9 = Refused"
# response coding for many questions. This maps that to 1 / 0 / NaN.
def _yes_no(series, yes_value=1, no_value=2):
    out = series.map({yes_value: 1, no_value: 0})
    return out


_BRFSS_AGE_MIDPOINT = {
    1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
    8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82,
}
_BRFSS_RACE_LABELS = {
    1: "White", 2: "Black", 3: "OtherRace", 4: "Multiracial", 5: "Hispanic",
}


def load_brfss(path=None):
    """
    BRFSS 2022 (Behavioral Risk Factor Surveillance System), a subset of
    columns extracted from the full annual file (`LLCP2022.XPT`, ~445k
    respondents, 328 columns) via `notebooks/build_brfss_subset.py`-style
    extraction. See that extraction step for the column list.

    Outcome: `_MICHD` -- CDC's computed "ever told you had coronary heart
    disease or myocardial infarction" variable. Coded 1 = yes, 2 = no in the
    raw data; we map to 1/0 and drop the ~1% of respondents with missing
    `_MICHD`.

    Protected attribute: `SEXVAR` (1 = male, 2 = female, as coded by BRFSS).
    `_RACEGR4` (race/ethnicity) is retained as a one-hot covariate block.

    Missing-data handling: BRFSS uses 7/9 (or 77/99, or 14) for "don't
    know" / "refused" / "missing" across different questions. Each variable
    is recoded to its natural scale here (fixed, deterministic maps only).
    Remaining missing values are median-imputed, but that imputation is NOT
    performed in this loader: the medians are data-derived statistics and are
    fitted on the training partition only, after the comparison-specific
    split, by src.preprocessing.TrainFittedPreprocessor. Missingness is
    ~1-11% per column depending on the question -- a limitation to note in
    the manuscript, not hidden.
    """
    df = pd.read_csv(_resolve_data_path(path, "brfss2022_subset.csv"))

    # --- Outcome: coronary heart disease / MI ---------------------------
    target_col = "heart_disease"
    df = df[df["_MICHD"].notna()].copy()
    df[target_col] = _yes_no(df["_MICHD"])

    # --- Protected attribute ---------------------------------------------
    group_col = "SEXVAR"
    group_labels = {1.0: "Male", 2.0: "Female"}

    # --- Demographics ------------------------------------------------------
    # _AGEG5YR: 1-13 = 5-year age bands from 18-24 up to 80+, 14 = missing.
    df["age_numeric"] = df["_AGEG5YR"].replace(14, np.nan).map(_BRFSS_AGE_MIDPOINT)

    # _RACEGR4: 1-5 = race/ethnicity groups, 9 = missing -> "Unknown".
    # Fixed code->label mapping only; the one-hot expansion is deferred to the
    # training-fitted preprocessor (src/preprocessing.py).
    df["race"] = df["_RACEGR4"].replace(9, np.nan).map(_BRFSS_RACE_LABELS).fillna("Unknown")

    # _INCOMG1 (1-7 income brackets) and _EDUCAG (1-4 education levels): 9 = missing.
    df["income_level"] = df["_INCOMG1"].replace(9, np.nan)
    df["education_level"] = df["_EDUCAG"].replace(9, np.nan)

    # --- Comorbidities / general health -----------------------------------
    df["diabetes_or_prediabetes"] = df["DIABETE4"].isin([1, 4]).astype(int)
    df["diabetes_or_prediabetes"] = df["diabetes_or_prediabetes"].where(df["DIABETE4"].isin([1, 2, 3, 4]), np.nan)

    for raw_col, new_col in [
        ("CVDSTRK3", "stroke"), ("ASTHMA3", "asthma"), ("CHCCOPD3", "copd"),
        ("CHCKDNY2", "kidney_disease"), ("HAVARTH4", "arthritis"),
        ("ADDEPEV3", "depression"), ("DIFFWALK", "diff_walking"),
    ]:
        df[new_col] = _yes_no(df[raw_col])

    df["bmi"] = df["_BMI5"] / 100.0

    # GENHLTH: 1 (excellent) - 5 (poor); 7/9 = missing.
    df["general_health"] = df["GENHLTH"].where(df["GENHLTH"].isin([1, 2, 3, 4, 5]), np.nan)

    # PHYSHLTH / MENTHLTH: 1-30 = days, 88 = none (-> 0), 77/99 = missing.
    for raw_col, new_col in [("PHYSHLTH", "phys_health_days"), ("MENTHLTH", "ment_health_days")]:
        df[new_col] = df[raw_col].replace(88, 0).where(df[raw_col].replace(88, 0) <= 30, np.nan)

    # --- Behavioral ---------------------------------------------------------
    # _TOTINDA: 1 = had physical activity in past 30 days, 2 = none, 9 = missing.
    df["physical_activity"] = df["_TOTINDA"].map({1: 1, 2: 0})
    # _RFDRHV8: 1 = not a heavy drinker, 2 = heavy drinker, 9 = missing.
    df["heavy_drinking"] = df["_RFDRHV8"].map({1: 0, 2: 1})
    df["smoked_100_cigs"] = _yes_no(df["SMOKE100"])

    # --- Access to care ------------------------------------------------------
    # _HLTHPLN: 1 = has health coverage, 2 = none, 9 = missing.
    df["has_health_coverage"] = df["_HLTHPLN"].map({1: 1, 2: 0})
    df["cost_barrier"] = _yes_no(df["MEDCOST1"])

    demographics_cols = ["age_numeric", "income_level", "education_level", "race"]
    comorbidity_cols = [
        "diabetes_or_prediabetes", "stroke", "asthma", "copd", "kidney_disease",
        "arthritis", "depression", "diff_walking", "bmi", "general_health",
        "phys_health_days", "ment_health_days",
    ]
    behavioral_cols = ["physical_activity", "heavy_drinking", "smoked_100_cigs"]
    access_cols = ["has_health_coverage", "cost_barrier"]

    feature_cols = demographics_cols + comorbidity_cols + behavioral_cols + access_cols

    # Median imputation is DEFERRED. It is a data-derived statistic and is
    # fitted on training rows only, after the comparison-specific split, by
    # src.preprocessing.TrainFittedPreprocessor.

    covariate_blocks = {
        "demographics": demographics_cols,
        "comorbidities": comorbidity_cols,
        "behavioral": behavioral_cols,
        "access": access_cols,
    }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "group_labels": group_labels,
        "covariate_blocks": covariate_blocks,
        "cluster_col": None,
        "preprocess_spec": {
            "categorical": [{"source": "race", "prefix": "race", "drop_first": True}],
        },
        "name": "BRFSS 2022 (Behavioral Risk Factor Surveillance System)",
    }


_NHANES_RACE_LABELS = {
    1: "MexicanAmerican", 2: "OtherHispanic", 3: "White",
    4: "Black", 6: "Asian", 7: "OtherMultiracial",
}


def load_nhanes(path=None):
    """
    NHANES 2017-2018 (National Health and Nutrition Examination Survey),
    restricted to adults (RIDAGEYR >= 18, n ~5,856), built by
    `scripts/fetch_nhanes.py` from 7 component files merged on `SEQN`.

    Outcome: `diabetes`, derived from `DIQ010` ("Has a doctor told you that
    you have diabetes?"): 1 = yes or 3 = borderline/prediabetes -> 1,
    2 = no -> 0 (consistent with the prediabetes+diabetes=positive
    convention used in dataset 1). The small number of "don't know" (9)
    responses are dropped.

    Protected attribute: `RIAGENDR` (1 = Male, 2 = Female).
    `RIDRETH3` (race/ethnicity) is retained as a one-hot covariate block.

    Feature engineering:
      - `RIDAGEYR` (age in years) used directly.
      - `RIDRETH3` -> one-hot (drop_first=True).
      - `DMDEDUC2` (education, 1-5; 7/9 = missing) and `INDFMPIR`
        (income-to-poverty ratio, continuous) -> median-imputed on training
        rows only (see src/preprocessing.py).
      - `BPQ020`/`BPQ080` (told high blood pressure / high cholesterol),
        `SMQ020` (smoked >=100 cigarettes), `PAQ605` (vigorous work
        activity), `HIQ011` (has health insurance) -> recoded 1/0, 9/7 ->
        NaN -> median-imputed.
      - `BMXBMI` (measured BMI) -> median-imputed (~7% missing).
    """
    df = pd.read_csv(_resolve_data_path(path, "nhanes_2017_2018.csv"))

    # Adults only: DIQ010, DMDEDUC2, etc. are adult-only questions.
    df = df[df["RIDAGEYR"] >= 18].copy()

    # --- Outcome -----------------------------------------------------------
    target_col = "diabetes"
    df = df[df["DIQ010"].isin([1, 2, 3])].copy()
    df[target_col] = df["DIQ010"].isin([1, 3]).astype(int)

    # --- Protected attribute -------------------------------------------------
    group_col = "RIAGENDR"
    group_labels = {1.0: "Male", 2.0: "Female"}

    # --- Demographics ----------------------------------------------------------
    df["age_years"] = df["RIDAGEYR"]

    # Fixed code->label mapping only; one-hot expansion is deferred to the
    # training-fitted preprocessor (src/preprocessing.py).
    df["race"] = df["RIDRETH3"].map(_NHANES_RACE_LABELS).fillna("OtherMultiracial")

    df["education_level"] = df["DMDEDUC2"].where(df["DMDEDUC2"].isin([1, 2, 3, 4, 5]), np.nan)
    df["income_to_poverty_ratio"] = df["INDFMPIR"]

    # --- Comorbidities ------------------------------------------------------
    df["high_bp"] = _yes_no(df["BPQ020"])
    df["high_cholesterol"] = _yes_no(df["BPQ080"])
    df["bmi"] = df["BMXBMI"]

    # --- Behavioral ---------------------------------------------------------
    df["smoked_100_cigs"] = _yes_no(df["SMQ020"])
    df["vigorous_activity"] = _yes_no(df["PAQ605"])

    # --- Access to care --------------------------------------------------------
    df["has_health_insurance"] = _yes_no(df["HIQ011"])

    demographics_cols = [
        "age_years", "education_level", "income_to_poverty_ratio", "race",
    ]
    comorbidity_cols = ["high_bp", "high_cholesterol", "bmi"]
    behavioral_cols = ["smoked_100_cigs", "vigorous_activity"]
    access_cols = ["has_health_insurance"]

    feature_cols = demographics_cols + comorbidity_cols + behavioral_cols + access_cols

    # Median imputation (education ~5%, income ~14%, BMI ~7%, plus a handful
    # of "don't know" responses on the yes/no items) is DEFERRED and fitted on
    # training rows only after the comparison-specific split.

    covariate_blocks = {
        "demographics": demographics_cols,
        "comorbidities": comorbidity_cols,
        "behavioral": behavioral_cols,
        "access": access_cols,
    }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "group_labels": group_labels,
        "covariate_blocks": covariate_blocks,
        "cluster_col": None,
        "preprocess_spec": {
            "categorical": [{"source": "race", "prefix": "race", "drop_first": True}],
        },
        "name": "NHANES 2017-2018",
    }


def _yes_no_str(series, yes_value="Yes", no_value="No"):
    """Map a CCHS-style string Yes/No column to {1, 0}, leaving anything
    else (valid skip, don't know, refusal, not stated, NaN) as NaN."""
    return series.map({yes_value: 1, no_value: 0})


_CCHS_AGE_MIDPOINT = {
    "18 to 34 years": 26,
    "35 to 49 years": 42,
    "50 to 64 years": 57,
    "65 and older": 72,
}

_CCHS_EDUCATION_LEVEL = {
    "Less than secondary school graduation": 1,
    "Secondary school graduation, no post-secondary education": 2,
    "Post-secondary certificate/diploma / university degree": 3,
}

_CCHS_INCOME_LEVEL = {
    "No income or less than $20,000": 1,
    "$20,000 to $39,999": 2,
    "$40,000 to $59,999": 3,
    "$60,000 to $79,999": 4,
    "$80,000 or more": 5,
}


def load_cchs(path=None):
    """
    Canadian Community Health Survey (CCHS), 2019-2020 Annual Component PUMF.

    Source: Statistics Canada, via Borealis Data
    (https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP3/ZVCGBK).
    See `scripts/fetch_cchs.py` for the (manual download +) extraction steps
    used to produce `data/cchs_2019_2020_subset.csv` from the full PUMF.

    Restricted to respondents aged 18+ (CCHS also samples 12-17 year olds,
    who are excluded here for comparability with the other adult-only
    datasets).

    Outcome: `diabetes`, derived from `CCC_095` ("Has diabetes"):
    "Yes" -> 1, "No" -> 0; "valid skip" / "don't know" / "refusal" /
    "not stated" rows are dropped.

    Protected attribute: `DHH_SEX` ("Sex at birth"): "Male" / "Female".

    Province (`GEOGPRV`) is retained as a one-hot demographic covariate --
    CCHS does not collect a race/ethnicity variable comparable to the US
    surveys, so province serves as the closest available geographic/
    socioeconomic stratifier.
    """
    df = pd.read_csv(_resolve_data_path(path, "cchs_2019_2020_subset.csv"))

    df = df[df["DHHGAGE"] != "12 to 17 years"].copy()

    target_col = "diabetes"
    df = df[df["CCC_095"].isin(["Yes", "No"])].copy()
    df[target_col] = (df["CCC_095"] == "Yes").astype(int)

    group_col = "DHH_SEX"
    group_labels = {"Male": "Male", "Female": "Female"}

    df["age_years"] = df["DHHGAGE"].map(_CCHS_AGE_MIDPOINT)
    df["education_level"] = df["EHG2DVH3"].map(_CCHS_EDUCATION_LEVEL)
    df["income_level"] = df["INCDGHH"].map(_CCHS_INCOME_LEVEL)

    # Province one-hot expansion is deferred to the training-fitted
    # preprocessor (src/preprocessing.py); GEOGPRV is retained as the raw
    # categorical source column.

    df["high_bp"] = _yes_no_str(df["CCC_065"])
    df["high_cholesterol"] = _yes_no_str(df["CCC_075"])
    df["overweight_or_obese"] = (df["HWTDGBCC"] == "Overweight / Obese - Class I, II, III").astype(int)
    df.loc[df["HWTDGBCC"].isnull(), "overweight_or_obese"] = np.nan

    df["current_smoker"] = df["SMKDVSTY"].isin(
        ["Current daily smoker", "Current occasional smoker"]
    ).astype(int)
    df.loc[df["SMKDVSTY"].isnull(), "current_smoker"] = np.nan
    df["physical_activity_minutes"] = df["PAADVMVA"]

    df["has_regular_healthcare_provider"] = _yes_no_str(df["PHC_020"])

    demographics_cols = [
        "age_years", "education_level", "income_level", "GEOGPRV",
    ]
    comorbidity_cols = ["high_bp", "high_cholesterol", "overweight_or_obese"]
    behavioral_cols = ["current_smoker", "physical_activity_minutes"]
    access_cols = ["has_regular_healthcare_provider"]

    feature_cols = demographics_cols + comorbidity_cols + behavioral_cols + access_cols

    # Median imputation (education, income, BMI class, smoking status, and the
    # comorbidity/access yes-no items all carry a small share of "don't
    # know"/"refusal"/"not stated" responses) is DEFERRED and fitted on
    # training rows only after the comparison-specific split.

    covariate_blocks = {
        "demographics": demographics_cols,
        "comorbidities": comorbidity_cols,
        "behavioral": behavioral_cols,
        "access": access_cols,
    }

    return {
        "df": df,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "group_col": group_col,
        "group_labels": group_labels,
        "covariate_blocks": covariate_blocks,
        "cluster_col": None,
        "preprocess_spec": {
            "categorical": [
                {"source": "GEOGPRV", "prefix": "province", "drop_first": True}
            ],
        },
        "name": "CCHS 2019-2020",
    }
