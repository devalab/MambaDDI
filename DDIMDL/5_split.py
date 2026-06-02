import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
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
INPUT = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_negative_sampling.csv"
df = pd.read_csv(INPUT)

# ---------------------------
# 3️⃣ Class counts
# ---------------------------
class_counts = df["Y"].value_counts().sort_index()

# Rare classes = classes with too few samples for stable stratified split
rare_classes = class_counts[class_counts < 7].index.tolist()
normal_classes = class_counts[class_counts >= 7].index.tolist()

print("Rare classes (<7 samples):", rare_classes)
print("Normal classes (>=7 samples):", normal_classes)

rare_df = df[df["Y"].isin(rare_classes)].copy()
normal_df = df[df["Y"].isin(normal_classes)].copy()

# ---------------------------
# 4️⃣ Normal stratified split
# ---------------------------
train_df, temp_df = train_test_split(
    normal_df,
    test_size=0.30,
    stratify=normal_df["Y"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=2/3,   # 10% val, 20% test overall
    stratify=temp_df["Y"],
    random_state=SEED
)

# ---------------------------
# 5️⃣ Manually split rare classes
# ---------------------------
rare_train_parts = []
rare_val_parts = []
rare_test_parts = []

for cls in rare_classes:
    cls_df = rare_df[rare_df["Y"] == cls].sample(frac=1, random_state=SEED)

    n = len(cls_df)

    if n == 1:
        # Only 1 sample → put in test
        rare_test_parts.append(cls_df)

    elif n == 2:
        # 1 train, 1 test
        rare_train_parts.append(cls_df.iloc[:1])
        rare_test_parts.append(cls_df.iloc[1:])

    elif n == 3:
        # 1 train, 1 val, 1 test
        rare_train_parts.append(cls_df.iloc[:1])
        rare_val_parts.append(cls_df.iloc[1:2])
        rare_test_parts.append(cls_df.iloc[2:])

    else:
        # For 4–6 samples: force at least 1 in val and 1 in test
        rare_train_parts.append(cls_df.iloc[:-2])
        rare_val_parts.append(cls_df.iloc[-2:-1])
        rare_test_parts.append(cls_df.iloc[-1:])

# Concatenate rare splits
if rare_train_parts:
    rare_train_df = pd.concat(rare_train_parts, ignore_index=True)
    train_df = pd.concat([train_df, rare_train_df], ignore_index=True)

if rare_val_parts:
    rare_val_df = pd.concat(rare_val_parts, ignore_index=True)
    val_df = pd.concat([val_df, rare_val_df], ignore_index=True)

if rare_test_parts:
    rare_test_df = pd.concat(rare_test_parts, ignore_index=True)
    test_df = pd.concat([test_df, rare_test_df], ignore_index=True)

# Shuffle splits
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df   = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
test_df  = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------------------------
# 6️⃣ Oversample TRAIN only
# ---------------------------
target_ratio = 0.005  # 0.5%
train_oversampled_df = train_df.copy()

counts = train_oversampled_df["Y"].value_counts().sort_index()
total_size = len(train_oversampled_df)
target_min = int(np.ceil(total_size * target_ratio))

oversampled_parts = [train_oversampled_df]

for cls, count in counts.items():
    if count < target_min:
        cls_df = train_oversampled_df[train_oversampled_df["Y"] == cls]
        needed = target_min - count

        sampled_extra = cls_df.sample(
            n=needed,
            replace=True,
            random_state=SEED
        )
        oversampled_parts.append(sampled_extra)

train_oversampled_df = pd.concat(oversampled_parts, ignore_index=True)
train_oversampled_df = train_oversampled_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ---------------------------
# 7️⃣ Save CSVs
# ---------------------------
save_dir = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing"
os.makedirs(save_dir, exist_ok=True)

train_path = os.path.join(save_dir, "train_ddimdl.csv")
val_path   = os.path.join(save_dir, "val_ddimdl.csv")
test_path  = os.path.join(save_dir, "test_ddimdl.csv")

train_oversampled_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

# ---------------------------
# 8️⃣ Save summary
# ---------------------------
summary_path = os.path.join(save_dir, "5_split.txt")

def pct(n, total):
    return round((n / total) * 100, 3)

with open(summary_path, "w") as f:
    f.write("===== TRAIN/VAL/TEST SPLIT + OVERSAMPLING SUMMARY =====\n\n")
    f.write(f"GLOBAL RANDOM SEED USED: {SEED}\n\n")
    f.write(f"Total dataset size: {len(df)}\n")
    f.write(f"Number of classes: {df['Y'].nunique()}\n")
    f.write(f"Rare classes (<7 samples): {rare_classes}\n\n")

    f.write("----- SPLIT SIZES -----\n")
    f.write(f"Train before oversampling: {len(train_df)}\n")
    f.write(f"Train after oversampling : {len(train_oversampled_df)}\n")
    f.write(f"Validation: {len(val_df)}\n")
    f.write(f"Test: {len(test_df)}\n\n")

    f.write("----- CLASS DISTRIBUTION (TRAIN AFTER OVERSAMPLING) -----\n")
    total_train = len(train_oversampled_df)
    for cls, count in train_oversampled_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_train)}%)\n")

    f.write("\n----- CLASS DISTRIBUTION (VALIDATION SET) -----\n")
    total_val = len(val_df)
    for cls, count in val_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_val)}%)\n")

    f.write("\n----- CLASS DISTRIBUTION (TEST SET) -----\n")
    total_test = len(test_df)
    for cls, count in test_df["Y"].value_counts().sort_index().items():
        f.write(f"Class {cls}: {count} ({pct(count, total_test)}%)\n")

print("✔ Split completed successfully.")
print("✔ Train saved to:", train_path)
print("✔ Val saved to:", val_path)
print("✔ Test saved to:", test_path)
print("✔ Summary saved to:", summary_path)