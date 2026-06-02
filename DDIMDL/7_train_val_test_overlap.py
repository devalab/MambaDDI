import pandas as pd

BASE = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing"

train_df = pd.read_csv(f"{BASE}/train_ddimdl.csv")
val_df   = pd.read_csv(f"{BASE}/val_ddimdl.csv")
test_df  = pd.read_csv(f"{BASE}/test_ddimdl.csv")

def get_drug_set(df):
    return set(df["Drug1"]).union(set(df["Drug2"]))

train_drugs = get_drug_set(train_df)
val_drugs   = get_drug_set(val_df)
test_drugs  = get_drug_set(test_df)

# Overlaps
train_test_overlap = train_drugs & test_drugs
train_val_overlap  = train_drugs & val_drugs
val_test_overlap   = val_drugs & test_drugs

print(f"Train drugs: {len(train_drugs)}")
print(f"Val drugs  : {len(val_drugs)}")
print(f"Test drugs : {len(test_drugs)}")

print("\nOVERLAPS:")
print(f"Train ∩ Test: {len(train_test_overlap)}")
print(f"Train ∩ Val : {len(train_val_overlap)}")
print(f"Val ∩ Test  : {len(val_test_overlap)}")

print("\nPercentage overlap (Train-Test):",
      100 * len(train_test_overlap) / len(test_drugs))
