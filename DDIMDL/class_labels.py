#!/usr/bin/env python3
"""
Generate classLabel -> event_name mapping CSV
from a DDI dataset file.

Input columns expected:
- DrugBankID_A
- Drug_A_Name
- DrugBankID_B
- Drug_B_Name
- classLabel
- event_name
"""

import pandas as pd
import os

# =========================
# PATHS
# =========================
INPUT_CSV = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/DDIMDL_DrugBankIDs_with_names.csv"   # <-- change if needed
OUTPUT_CSV = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/class_label_mapping.csv"

# =========================
# LOAD DATA
# =========================
print("Loading file...")
df = pd.read_csv(INPUT_CSV)

print(f"Total rows in input: {len(df)}")

# =========================
# CHECK REQUIRED COLUMNS
# =========================
required_cols = ["classLabel", "event_name"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================
# EXTRACT UNIQUE MAPPING
# =========================
mapping_df = df[["classLabel", "event_name"]].drop_duplicates()

# sort by classLabel
mapping_df = mapping_df.sort_values("classLabel").reset_index(drop=True)

# =========================
# CHECK FOR CONFLICTS
# =========================
# If one classLabel maps to multiple event_names, flag it
conflicts = (
    df.groupby("classLabel")["event_name"]
    .nunique()
    .reset_index()
)

conflicts = conflicts[conflicts["event_name"] > 1]

if len(conflicts) > 0:
    print("\nWARNING: Some classLabels map to multiple event names!")
    print(conflicts)

    print("\nDetailed conflicting mappings:")
    conflict_details = (
        df[["classLabel", "event_name"]]
        .drop_duplicates()
        .sort_values("classLabel")
    )
    print(conflict_details[conflict_details["classLabel"].isin(conflicts["classLabel"])])

else:
    print("No conflicts found. Each classLabel maps cleanly to one event_name.")

# =========================
# SAVE OUTPUT
# =========================
mapping_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved mapping CSV to:\n{OUTPUT_CSV}")
print(f"Total unique class labels: {len(mapping_df)}")

print("\nPreview:")
print(mapping_df.head(20))