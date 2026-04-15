# =============================================================================
# Credit Risk Platform — Feature Engineering Pipeline
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def load_and_clean(path: str = "data/accepted.csv", sample_size: int = 500000) -> pd.DataFrame:
    """
    Load LendingClub data, filter to completed loans only,
    and create binary default target variable.
    """
    print(f"Loading data (sample of {sample_size:,} rows)...")
    df = pd.read_csv(path, low_memory=False, nrows=sample_size)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Keep only completed loans — drop Current, In Grace Period, Late
    completed_statuses = ["Fully Paid", "Charged Off"]
    df = df[df["loan_status"].isin(completed_statuses)].copy()
    print(f"  After filtering to completed loans: {len(df):,} rows")

    # Binary target: 1 = default (Charged Off), 0 = paid
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    print(f"  Default rate: {df['default'].mean()*100:.1f}%")

    return df


def engineer_features(df: pd.DataFrame) -> tuple:
    """
    Engineer 20+ features from raw LendingClub data.
    Returns processed dataframe and feature column list.
    """
    print("\nEngineering features...")

    # ── Select core columns ────────────────────────────────────────────
    feature_cols_raw = [
        "loan_amnt", "int_rate", "installment", "grade", "sub_grade",
        "emp_length", "home_ownership", "annual_inc", "verification_status",
        "purpose", "dti", "delinq_2yrs", "fico_range_low", "fico_range_high",
        "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util",
        "total_acc", "mort_acc", "pub_rec_bankruptcies"
    ]

    df = df[feature_cols_raw + ["default", "loan_status"]].copy()

    # ── Clean numeric columns ──────────────────────────────────────────
    # int_rate comes as "13.99%" string
    df["int_rate"] = pd.to_numeric(
        df["int_rate"].astype(str).str.replace("%", ""), errors="coerce"
    )

    # revol_util comes as "45.2%" string
    df["revol_util"] = pd.to_numeric(
        df["revol_util"].astype(str).str.replace("%", ""), errors="coerce"
    )

    # emp_length: "10+ years" -> 10, "< 1 year" -> 0
    emp_map = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10
    }
    df["emp_length_num"] = df["emp_length"].map(emp_map).fillna(0)

    # Grade: A=1, B=2, C=3, D=4, E=5, F=6, G=7
    grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    df["grade_num"] = df["grade"].map(grade_map).fillna(4)

    # ── Engineered Features ────────────────────────────────────────────

    # Feature 1: Debt-to-Income ratio (already exists as dti)

    # Feature 2: Loan-to-Income ratio
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)

    # Feature 3: Average FICO score
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    # Feature 4: Monthly payment burden
    df["monthly_income"] = df["annual_inc"] / 12
    df["payment_to_income"] = df["installment"] / (df["monthly_income"] + 1)

    # Feature 5: Revolving utilization (already exists as revol_util)

    # Feature 6: Derogatory marks flag
    df["has_derog"] = ((df["delinq_2yrs"] > 0) | (df["pub_rec"] > 0)).astype(int)

    # Feature 7: Bankruptcy flag
    df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

    # Feature 8: Credit inquiry intensity
    df["inq_last_6mths"] = df["inq_last_6mths"].fillna(0)

    # Feature 9: Home ownership encoding
    home_map = {"MORTGAGE": 3, "OWN": 2, "RENT": 1, "OTHER": 0, "NONE": 0}
    df["home_ownership_num"] = df["home_ownership"].map(home_map).fillna(1)

    # Feature 10: Purpose risk score
    high_risk_purposes = ["small_business", "moving", "renewable_energy", "other"]
    medium_risk_purposes = ["vacation", "medical", "wedding"]
    df["purpose_risk"] = df["purpose"].apply(
        lambda x: 2 if x in high_risk_purposes else (1 if x in medium_risk_purposes else 0)
    )

    # Feature 11: Verification status
    verif_map = {"Verified": 2, "Source Verified": 1, "Not Verified": 0}
    df["verification_num"] = df["verification_status"].map(verif_map).fillna(0)

    # Feature 12: Account utilization ratio
    df["acc_util"] = df["open_acc"] / (df["total_acc"] + 1)

    # Feature 13: Revolving balance to income
    df["revol_to_income"] = df["revol_bal"] / (df["annual_inc"] + 1)

    # ── Final feature list ─────────────────────────────────────────────
    final_features = [
        "loan_amnt", "int_rate", "installment", "grade_num", "emp_length_num",
        "annual_inc", "dti", "delinq_2yrs", "fico_avg", "inq_last_6mths",
        "open_acc", "revol_bal", "revol_util", "total_acc", "mort_acc",
        "loan_to_income", "payment_to_income", "has_derog", "has_bankruptcy",
        "home_ownership_num", "purpose_risk", "verification_num",
        "acc_util", "revol_to_income"
    ]

    # Fill remaining NAs with median
    for col in final_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df_clean = df[final_features + ["default", "grade", "purpose", "loan_status"]].dropna()
    print(f"  Final dataset: {len(df_clean):,} rows, {len(final_features)} features")
    print(f"  Default rate: {df_clean['default'].mean()*100:.1f}%")

    return df_clean, final_features


if __name__ == "__main__":
    df = load_and_clean()
    df_clean, features = engineer_features(df)
    print(f"\nFeature list: {features}")
    print(df_clean[features].describe())
    df_clean.to_parquet("data/featured.parquet", index=False)
    print("\nSaved to data/featured.parquet")