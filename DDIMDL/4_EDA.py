import pandas as pd

# Load your CSV
df = pd.read_csv("/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/ddimdl_negative_sampling.csv")

# --- Class statistics ---
num_classes = df["Y"].nunique()
class_counts = df["Y"].value_counts()
class_counts_sorted = df["Y"].value_counts().sort_index()

# Percentage contribution of each class
class_percent = (class_counts / len(df) * 100).round(2)

# --- Drug statistics ---
# Count unique drugs appearing in either column
unique_drugs = pd.concat([df["Drug1_ID"], df["Drug2_ID"]]).nunique()

# Total number of interaction pairs = number of rows
num_pairs = len(df)

# --- Prepare text content ---
output_text = []
output_text.append(f"Number of classes: {num_classes}\n")

output_text.append("Samples per class:\n")
output_text.append(class_counts.to_string() + "\n\n")

output_text.append("Percentage contribution of each class:\n")
output_text.append(class_percent.to_string() + "\n\n")

output_text.append("Samples per class (sorted by class label):\n")
output_text.append(class_counts_sorted.to_string() + "\n\n")

output_text.append(f"Number of unique drugs: {unique_drugs}\n")
output_text.append(f"Number of drug interaction pairs: {num_pairs}\n")

# --- Save to text file ---
with open("/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/4_EDA.txt", "w") as file:
    file.write("\n".join(output_text))

print("Saved results!")
