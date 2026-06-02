import pandas as pd

# =========================
# PATHS
# =========================
DRKG_TSV = "/home2/jessygrace.polinati/Abhi/dd/drkg/drkg.tsv"
DDI_CSV  = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_rdkit_filtered.csv"

# =========================
# LOAD DRKG
# =========================
print("Loading DRKG...")
drkg = pd.read_csv(
    DRKG_TSV,
    sep="\t",
    header=None,
    names=["head", "relation", "tail"],
    dtype=str
)

# Extract DrugBank drugs from DRKG
def extract_drugs(series):
    return series[series.str.startswith("Compound::DB", na=False)] \
        .str.replace("Compound::", "", regex=False)

drkg_drugs = set()
drkg_drugs.update(extract_drugs(drkg["head"]))
drkg_drugs.update(extract_drugs(drkg["tail"]))

print(f"Unique DrugBank drugs in DRKG: {len(drkg_drugs)}")

# =========================
# LOAD DrugBank DDI DATASET
# =========================
print("Loading DrugBank DDI dataset...")
ddi = pd.read_csv(DDI_CSV, dtype=str)

ddi_drugs = set(ddi["Drug1_ID"]).union(set(ddi["Drug2_ID"]))

print(f"Unique DrugBank drugs in DDI dataset: {len(ddi_drugs)}")

# =========================
# OVERLAP
# =========================
#print(ddi_drugs)
overlap = drkg_drugs.intersection(ddi_drugs)

print("\n========== OVERLAP ==========")
print(f"Overlapping drugs: {len(overlap)}")
print(f"Coverage: {100 * len(overlap) / len(ddi_drugs):.2f}%")

# =========================
# SAMPLE
# =========================
print("\nSample overlapping drugs:")
print(list(overlap)[:10])
