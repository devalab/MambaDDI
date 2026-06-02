import pandas as pd
import os

# ---------------------------
# Paths
# ---------------------------
INPUT = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_negative_sampling.csv"
OUTPUT = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_negative_sampling.csv"
SUMMARY = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/label_fix_summary.txt"

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(INPUT)

print("Original unique labels:")
print(sorted(df["Y"].unique()))

# ---------------------------
# Build continuous remap
# ---------------------------
old_labels = sorted(df["Y"].unique())
label_map = {old: new for new, old in enumerate(old_labels)}

# Apply remap
df["Y"] = df["Y"].map(label_map).astype(int)

print("\nNew unique labels:")
print(sorted(df["Y"].unique()))

# ---------------------------
# Save fixed CSV
# ---------------------------
df.to_csv(OUTPUT, index=False)

# ---------------------------
# Save summary
# ---------------------------
with open(SUMMARY, "w") as f:
    f.write("===== LABEL FIX SUMMARY =====\n\n")
    f.write(f"Input file: {INPUT}\n")
    f.write(f"Output file: {OUTPUT}\n\n")
    f.write("Old -> New label mapping:\n")
    for old, new in label_map.items():
        f.write(f"{old} -> {new}\n")
    f.write("\n")
    f.write(f"Final number of classes: {df['Y'].nunique()}\n")
    f.write(f"Min label: {df['Y'].min()}\n")
    f.write(f"Max label: {df['Y'].max()}\n")

print("\n✔ Fixed dataset saved to:", OUTPUT)
print("✔ Summary saved to:", SUMMARY)
print("✔ Final number of classes:", df["Y"].nunique())
print("✔ Final labels:", sorted(df["Y"].unique()))