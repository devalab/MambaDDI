import pandas as pd
import numpy as np

TRAIN_CSV = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/train_ddimdl.csv"
VAL_CSV   = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/val_ddimdl.csv"
TEST_CSV  = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/test_ddimdl.csv"

RARE_THRESHOLD = 50   # you can change this later

def analyze_split(name, df):
    print(f"\n===== {name.upper()} SPLIT =====")
    total = len(df)
    counts = df["Y"].value_counts().sort_index()

    summary = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "percentage": (counts.values / total) * 100
    })

    print(f"Total samples: {total}")
    print(summary)

    rare_classes = summary[summary["count"] < RARE_THRESHOLD]["class"].tolist()
    print(f"\nRare classes (< {RARE_THRESHOLD} samples):")
    print(rare_classes)

    return set(summary["class"]), set(rare_classes)

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Analyze
train_classes, train_rare = analyze_split("train", train_df)
val_classes, val_rare     = analyze_split("val", val_df)
test_classes, test_rare   = analyze_split("test", test_df)

# =========================
# Cross-split consistency
# =========================
print("\n===== CROSS-SPLIT CHECKS =====")

print("Classes in train but not in val:", sorted(train_classes - val_classes))
print("Classes in train but not in test:", sorted(train_classes - test_classes))

print("\nRare classes in TRAIN missing in VAL:", sorted(train_rare - val_classes))
print("Rare classes in TRAIN missing in TEST:", sorted(train_rare - test_classes))

print("\nRare classes common across ALL splits:",
      sorted(train_rare & val_rare & test_rare))
