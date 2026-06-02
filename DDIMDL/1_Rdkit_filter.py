import pandas as pd
from rdkit import Chem

# Load original dataset
df = pd.read_csv("/home2/jessygrace.polinati/Abhi/dd/DDIMDL/filtered_drug_dataset.csv")

# Function to check if SMILES is valid
def is_valid_smiles(s):
    return Chem.MolFromSmiles(s) is not None

# Filter rows where both Drug1 and Drug2 SMILES can be parsed
filtered_df = df[df["Drug1"].apply(is_valid_smiles) & df["Drug2"].apply(is_valid_smiles)]

# Count unique valid drugs
unique_drugs = pd.concat([filtered_df["Drug1"], filtered_df["Drug2"]]).nunique()

# Count valid interaction pairs (rows)
num_pairs = len(filtered_df)

# Save filtered dataset
output_path = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_rdkit_filtered.csv"
filtered_df.to_csv(output_path, index=False)

print("Filtered dataset saved to:", output_path)
print("Number of unique valid drugs:", unique_drugs)
print("Number of valid interaction pairs:", num_pairs)
