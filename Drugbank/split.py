import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import math
import os

# ---------------------------
# 1️⃣ Set global seed
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 2️⃣ Load dataset
# ---------------------------
INPUT = "/home2/jessygrace.polinati/Abhi/dd/DrugBank_DDI_with_negatives.csv"
df = pd.read_csv(INPUT)

# ---------------------------
# 3️⃣ Stratified Train/Val/Test split (70/10/20)
# ---------------------------
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["Y"], random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df, test_size=2/3, stratify=temp_df["Y"], random_state=SEED
)

# ---------------------------
# 4️⃣ Oversample train set so each class >= 0.5% of final train
# ---------------------------
target_ratio = 0.005  # 0.5%
train_oversampled_df = train_df.copy()

# Iteratively oversample small classes
while True:
    total_size = len(train_oversampled_df)
    counts = train_oversampled_df["Y"].value_counts()
    min_needed = total_size * target_ratio
    below_min = counts[counts < min_needed]
    
    if below_min.empty:
        break  # all classes >= 0.5%
    
    # Duplicate all underrepresented classes once
    new_rows = [train_oversampled_df[train_oversampled_df["Y"] == cls] for cls in below_min.index]
    train_oversampled_df = pd.concat([train_oversampled_df] + new_rows, ignore_index=True)

# Shuffle train set reproducibly
train_oversampled_df = train_oversampled_df.sample(frac=1, random_state=SEED)

# ---------------------------
# 5️⃣ Save CSVs
# ---------------------------
save_dir = "/home2/jessygrace.polinati/Abhi/dd/"
os.makedirs(save_dir, exist_ok=True)

train_path = os.path.join(save_dir, "train_ddi.csv")
val_path   = os.path.join(save_dir, "val_ddi.csv")
test_path  = os.path.join(save_dir, "test_ddi.csv")

train_oversampled_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# ---------------------------
# 6️⃣ Save summary TXT
# ---------------------------
summary_path = os.path.join(save_dir, "split_oversampling_summary.txt")

def pct(n, total):
    return round((n / total) * 100, 3)

with open(summary_path, "w") as f:
    f.write("===== TRAIN/VAL/TEST SPLIT + OVERSAMPLING SUMMARY =====\n\n")
    
    f.write(f"GLOBAL RANDOM SEED USED: {SEED}\n\n")
    
    f.write("----- SPLIT SIZES -----\n")
    f.write(f"Train before oversampling: {len(train_df)}\n")
    f.write(f"Train after oversampling : {len(train_oversampled_df)}\n")
    f.write(f"Validation: {len(val_df)}\n")
    f.write(f"Test: {len(test_df)}\n\n")
    
    # TRAIN BEFORE oversampling
    f.write("----- CLASS DISTRIBUTION (TRAIN BEFORE OVERSAMPLING) -----\n")
    total_train_before = len(train_df)
    for cls, count in train_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_train_before)}%)\n")
    
    # TRAIN AFTER oversampling
    f.write("\n----- CLASS DISTRIBUTION (TRAIN AFTER OVERSAMPLING) -----\n")
    total_train_after = len(train_oversampled_df)
    for cls, count in train_oversampled_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_train_after)}%)\n")
    
    f.write(f"\nOversampling ensured all classes ≥ {target_ratio*100}% contribution.\n\n")
    
    # VAL distribution
    f.write("----- CLASS DISTRIBUTION (VALIDATION SET) -----\n")
    total_val = len(val_df)
    for cls, count in val_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_val)}%)\n")
    
    # TEST distribution
    f.write("\n----- CLASS DISTRIBUTION (TEST SET) -----\n")
    total_test = len(test_df)
    for cls, count in test_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_test)}%)\n")

print("✔ Train/Val/Test split with oversampling completed.")
print("✔ CSVs saved at:", save_dir)
print("✔ Summary saved at:", summary_path)
