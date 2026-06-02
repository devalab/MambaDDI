import pandas as pd
import random

# Load the filtered dataset (this contains both Drug IDs → SMILES)
df = pd.read_csv("/home2/jessygrace.polinati/Abhi/dd/DrugBank_DDI_filtered.csv")

# ---------------------------------------------
# 1️⃣ Build DrugBankID → SMILES mapping
# ---------------------------------------------
smiles_map = {}

# Collect mapping from Drug1_ID → Drug1
for d_id, smi in zip(df["Drug1_ID"], df["Drug1"]):
    if pd.notna(smi) and smi != "":
        smiles_map[d_id] = smi

# Collect mapping from Drug2_ID → Drug2
for d_id, smi in zip(df["Drug2_ID"], df["Drug2"]):
    if pd.notna(smi) and smi != "":
        smiles_map[d_id] = smi

print("Total drugs with known SMILES:", len(smiles_map))

# ---------------------------------------------
# 2️⃣ Prepare negative sampling
# ---------------------------------------------
num_pairs = len(df)

all_drugs = pd.concat([df["Drug1_ID"], df["Drug2_ID"]]).unique().tolist()

# set of positive real DDIs
positive_pairs = set()
for _, row in df.iterrows():
    pair = tuple(sorted((row["Drug1_ID"], row["Drug2_ID"])))
    positive_pairs.add(pair)

# negatives to generate (1:1)
num_neg = num_pairs
negative_pairs = set()

# ---------------------------------------------
# 3️⃣ Generate negative DDI pairs
# ---------------------------------------------
while len(negative_pairs) < num_neg:
    d1, d2 = random.sample(all_drugs, 2)
    pair = tuple(sorted((d1, d2)))

    if pair not in positive_pairs and pair not in negative_pairs:
        negative_pairs.add(pair)

# ---------------------------------------------
# 4️⃣ Convert negative pairs to DataFrame + fill SMILES
# ---------------------------------------------
negative_rows = []

for d1, d2 in negative_pairs:
    negative_rows.append({
        "Drug1_ID": d1,
        "Drug1": smiles_map.get(d1, ""),   # FILL SMILES HERE
        "Drug2_ID": d2,
        "Drug2": smiles_map.get(d2, ""),   # FILL SMILES HERE
        "Y": 0
    })

negative_df = pd.DataFrame(negative_rows)

# ---------------------------------------------
# 5️⃣ Combine positive + negative
# ---------------------------------------------
final_df = pd.concat([df, negative_df], ignore_index=True)

# ---------------------------------------------
# 6️⃣ Save dataset
# ---------------------------------------------
csv_path = "/home2/jessygrace.polinati/Abhi/dd/DrugBank_DDI_with_negatives.csv"
final_df.to_csv(csv_path, index=False)

# ---------------------------------------------
# 7️⃣ Save summary TXT
# ---------------------------------------------
txt_path = "/home2/jessygrace.polinati/Abhi/dd/negative_sampling_full_summary.txt"

summary = []
summary.append("FULL NEGATIVE SAMPLING SUMMARY\n")
summary.append("---------------------------------\n")
summary.append(f"Original classes (positive only): {df['Y'].nunique()}\n")
summary.append(f"NEW class added: 0 (negative pairs)\n")
summary.append(f"Final number of classes: {final_df['Y'].nunique()}\n\n")
summary.append(f"Original DDI pairs: {num_pairs}\n")
summary.append(f"Negative samples generated: {len(negative_df)}\n")
summary.append(f"Final dataset size: {len(final_df)}\n")

with open(txt_path, "w") as f:
    f.write("".join(summary))

print("✔ Negative sampling completed.")
print("✔ Final dataset saved to:", csv_path)
print("✔ Summary saved to:", txt_path)
