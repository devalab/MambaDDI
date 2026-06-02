#!/usr/bin/env python3
"""
Experiment C: Reverse Cross-Dataset Transfer (STRICT)

Train: DDIMDL
Val:   DDIMDL
Test:  DDI

Goal:
- Train on DDIMDL
- Evaluate on DDI
- Use ONLY semantically shared classes between datasets

This is a strict and fair cross-dataset transfer setup.
"""

import os, time, random, logging, copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.utils import add_self_loops

try:
    from mamba_ssm import Mamba2
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    raise ImportError("mamba_ssm not installed. Please install it first.")

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# =========================
# SEEDS & DEVICE
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PATHS
# =========================
# DDIMDL train/val
TRAIN_CSV = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/train_ddimdl.csv"
VAL_CSV   = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/preprocessing/val_ddimdl.csv"

# DDI test
TEST_CSV  = "/home2/jessygrace.polinati/Abhi/dd/train_ddi.csv"

DRKG_ENTITIES = "/home2/jessygrace.polinati/Abhi/dd/drkg/embed/entities.tsv"
DRKG_EMB      = "/home2/jessygrace.polinati/Abhi/dd/drkg/embed/DRKG_TransE_l2_entity.npy"

OUT_DIR = "/home2/jessygrace.polinati/Abhi/dd/DDIMDL/models/expC_ddimdl_to_ddi_strict"
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_TXT   = os.path.join(OUT_DIR, "expC_strict_summary.txt")
BEST_MODEL_PT = os.path.join(OUT_DIR, "mambaddi_expC_best.pt")
CLASSMAP_CSV  = os.path.join(OUT_DIR, "shared_class_mapping.csv")

# =========================
# HYPERPARAMS
# =========================
GNN_HIDDEN = 128
GNN_OUT = 64
DRKG_DIM = 400
FUSED_DIM = GNN_OUT + DRKG_DIM

BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 20
D_MODEL = 256
TAU = 0.65
PATIENCE = 5

LAMBDA_GENERAL = 0.2
LAMBDA_RARE = 0.4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("MambaDDI_ExperimentC")

# =========================================================
# STRICT SEMANTIC CLASS ALIGNMENT
# DDIMDL label -> DDI label
# =========================================================
DDIMDL_TO_DDI = {
    0: 49,   # adverse effects increase
    1: 75,   # serum concentration decrease
    2: 73,   # serum concentration increase
    3: 47,   # metabolism decrease
    4: 77,   # active metabolites increase
    5: 11,   # active metabolites reduce
    6: 58,   # cardiotoxic activities decrease
    7: 37,   # antihypertensive activities decrease
    8: 70,   # therapeutic efficacy decrease
    9: 9,    # hypoglycemic activities increase
    10: 54,  # bradycardic activities increase
    11: 60,  # hypotensive activities increase
    12: 14,  # bronchodilatory activities decrease
    13: 8,   # therapeutic efficacy increase
    14: 25,  # AV block increase
    15: 71,  # hypertensive activities increase
    16: 10,  # antihypertensive activities increase
    17: 19,  # vasoconstricting activities increase
    18: 30,  # orthostatic hypotensive activities increase
    19: 85,  # tachycardic activities increase
    20: 6,   # anticoagulant activities increase
    21: 72,  # excretion decrease -> higher serum level
    22: 4,   # metabolism increase
    23: 57,  # nephrotoxic activities increase
    24: 66,  # bleeding increase
    25: 68,  # hyperkalemic activities increase
    26: 27,  # neuroexcitatory activities increase
    27: 65,  # excretion increase -> lower serum level
    28: 12,  # anticoagulant activities decrease
    29: 80,  # hepatotoxic activities increase
    30: 56,  # hypotension increase
    31: 53,  # antiplatelet activities increase
    32: 2,   # anticholinergic activities increase
    33: 16,  # CNS depressant increase
    35: 20,  # QTc-prolonging activities increase
    36: 82,  # arrhythmogenic activities increase
    37: 15,  # cardiotoxic activities increase
    38: 83,  # hypokalemic activities increase
    39: 74,  # fluid retaining activities increase
    40: 69,  # analgesic activities increase
    41: 64,  # serotonergic activities increase
    42: 5,   # vasoconstricting activities decrease
    43: 76,  # sedative activities decrease
    45: 42,  # hyperkalemia risk increase
    46: 33,  # QTc prolongation risk increase
    47: 84,  # vasopressor activities increase
    48: 61,  # stimulatory activities decrease
    49: 18,  # increased absorption
    50: 67,  # decreased absorption
    51: 24,  # hypocalcemic activities increase
    52: 40,  # respiratory depressant increase
    53: 34,  # immunosuppressive activities increase
    55: 38,  # vasodilatory activities increase
    57: 48,  # myopathic rhabdomyolysis activities increase
    60: 63,  # myelosuppressive activities increase
    61: 78,  # hyperglycemic activities increase
    62: 55,  # hyponatremic activities increase
    63: 17,  # neuromuscular blocking decrease
    64: 81,  # thrombogenic activities increase
}

# shared DDIMDL labels
SHARED_DDIMDL_LABELS = sorted(list(DDIMDL_TO_DDI.keys()))
SHARED_DDI_LABELS = sorted(list(set(DDIMDL_TO_DDI.values())))

# reindex into contiguous 0..K-1
DDIMDL_SHARED_TO_NEW = {old: i for i, old in enumerate(SHARED_DDIMDL_LABELS)}
DDI_SHARED_TO_NEW = {old: DDIMDL_SHARED_TO_NEW[ddimdl_lab]
                     for ddimdl_lab, old in DDIMDL_TO_DDI.items()}

NUM_SHARED_CLASSES = len(SHARED_DDIMDL_LABELS)

# =========================
# ATOM FEATURES
# =========================
def atom_features(atom):
    atom_type = [0]*64
    atom_type[min(atom.GetAtomicNum(), 63)] = 1

    chiral = [0]*7
    chiral[min(int(atom.GetChiralTag()), 6)] = 1

    degree = [0]*11
    degree[min(atom.GetDegree(), 10)] = 1

    fc = [0]*12
    idx = atom.GetFormalCharge() + 6
    if 0 <= idx < 12:
        fc[idx] = 1

    nH = [0]*5
    nH[min(atom.GetTotalNumHs(), 4)] = 1

    hybrid = [0]*7
    hybrid[min(int(atom.GetHybridization()), 6)] = 1

    aromatic = int(atom.GetIsAromatic())
    ring = int(atom.IsInRing())

    return torch.tensor(
        atom_type + chiral + degree + fc + nH + hybrid + [aromatic] + [ring],
        dtype=torch.float32
    )

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    if len(atoms) == 0:
        return None

    x = torch.stack([atom_features(a) for a in atoms])

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges += [[i, j], [j, i]]

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
    return Data(x=x, edge_index=edge_index)

# =========================
# GAT ENCODER
# =========================
class GATNet(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, heads=[4,4,4]):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden, heads=heads[0], dropout=0.1)
        self.gat2 = GATConv(hidden*heads[0], hidden, heads=heads[1], dropout=0.1)
        self.gat3 = GATConv(hidden*heads[1], out_dim, heads=heads[2], concat=False, dropout=0.1)

    def forward(self, x, edge_index, batch):
        h1 = F.elu(self.gat1(x, edge_index))
        h2 = F.elu(self.gat2(h1, edge_index))
        h3_ = self.gat3(h2, edge_index)
        h3 = h3_ + h2[:, :h3_.size(1)]
        return global_add_pool(h3, batch)

# =========================
# MAMBA EXPERTS
# =========================
class GeneralExpert(nn.Module):
    def __init__(self, input_dim, n_class, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls_token.expand(B, 1, -1)

        seq_ab = torch.cat([cls, x], dim=1)
        out_ab = self.mamba(seq_ab)
        if isinstance(out_ab, tuple):
            out_ab = out_ab[0]
        rep_ab = out_ab[:, 0]

        x_rev = torch.flip(x, dims=[1])
        seq_ba = torch.cat([cls, x_rev], dim=1)
        out_ba = self.mamba(seq_ba)
        if isinstance(out_ba, tuple):
            out_ba = out_ba[0]
        rep_ba = out_ba[:, 0]

        rep = 0.5 * (rep_ab + rep_ba)
        rep = self.norm(rep)
        logits = self.head(rep)
        return logits, rep

class RareExpert(nn.Module):
    def __init__(self, input_dim, n_class, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.mamba = Mamba2(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2
        )

        self.norm1 = nn.LayerNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_class)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls_token.expand(B, 1, -1)

        seq_ab = torch.cat([cls, x], dim=1)
        out_ab = self.mamba(seq_ab)
        if isinstance(out_ab, tuple):
            out_ab = out_ab[0]
        out_ab = self.norm1(out_ab)
        out_ab = self.transformer(out_ab)
        rep_ab = out_ab[:, 0]

        x_rev = torch.flip(x, dims=[1])
        seq_ba = torch.cat([cls, x_rev], dim=1)
        out_ba = self.mamba(seq_ba)
        if isinstance(out_ba, tuple):
            out_ba = out_ba[0]
        out_ba = self.norm1(out_ba)
        out_ba = self.transformer(out_ba)
        rep_ba = out_ba[:, 0]

        rep = 0.5 * (rep_ab + rep_ba)
        rep = self.norm2(rep)
        logits = self.head(rep)
        return logits, rep

# =========================
# CONFIDENCE-AWARE GATING
# =========================
class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, 2)

    def forward(self, embed_g, embed_r, logits_g, logits_r):
        probs_g = F.softmax(logits_g, dim=1)
        probs_r = F.softmax(logits_r, dim=1)

        conf_g = probs_g.max(dim=1, keepdim=True).values
        conf_r = probs_r.max(dim=1, keepdim=True).values

        concat = torch.cat([embed_g, embed_r, probs_g, probs_r, conf_g, conf_r], dim=1)
        x = F.relu(self.fc1(concat))
        gate = F.softmax(self.fc2(x), dim=1)
        return gate, conf_g, conf_r

# =========================
# FULL MODEL
# =========================
class MambaDDIModel(nn.Module):
    def __init__(self, node_dim, drkg_matrix, n_class):
        super().__init__()
        self.gnn = GATNet(node_dim, GNN_HIDDEN, GNN_OUT)
        self.drkg_matrix = drkg_matrix

        self.general_expert = GeneralExpert(FUSED_DIM, n_class)
        self.rare_expert = RareExpert(FUSED_DIM, n_class)

        gate_input_dim = D_MODEL*2 + n_class*2 + 2
        self.gating = GatingNetwork(gate_input_dim)

    def encode_drugs(self, batch_graphs, drug_indices):
        gnn_vec = self.gnn(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
        kg_vec = self.drkg_matrix[drug_indices]
        fused = torch.cat([gnn_vec, kg_vec], dim=1)
        return fused

    def forward(self, g1_batch, g2_batch, idx1, idx2):
        emb1 = self.encode_drugs(g1_batch, idx1)
        emb2 = self.encode_drugs(g2_batch, idx2)

        pair_x = torch.stack([emb1, emb2], dim=1)

        logits_g, embed_g = self.general_expert(pair_x)
        logits_r, embed_r = self.rare_expert(pair_x)

        gate, conf_g, conf_r = self.gating(embed_g, embed_r, logits_g, logits_r)

        low_conf_mask = (conf_g < TAU).float()
        gate_adjust = gate.clone()
        gate_adjust[:, 1] = gate_adjust[:, 1] + 0.15 * low_conf_mask.squeeze(1)
        gate_adjust = gate_adjust / gate_adjust.sum(dim=1, keepdim=True)

        logits = (
            gate_adjust[:, 0].unsqueeze(1) * logits_g +
            gate_adjust[:, 1].unsqueeze(1) * logits_r
        )

        return logits, logits_g, logits_r, gate_adjust

# =========================
# DATASET
# =========================
class DDIDataset(Dataset):
    def __init__(self, df, smiles_to_graph, smiles_to_idx, source="ddimdl"):
        self.samples = []

        for _, row in df.iterrows():
            d1 = row["Drug1"]
            d2 = row["Drug2"]
            y  = int(row["Y"])

            mapped = self.map_label(y, source)
            if mapped is None:
                continue

            if d1 in smiles_to_graph and d2 in smiles_to_graph:
                self.samples.append((d1, d2, mapped))

        self.smiles_to_graph = smiles_to_graph
        self.smiles_to_idx = smiles_to_idx

    def map_label(self, y, source):
        if source == "ddimdl":
            if y in DDIMDL_SHARED_TO_NEW:
                return DDIMDL_SHARED_TO_NEW[y]
            return None

        elif source == "ddi":
            if y in DDI_SHARED_TO_NEW:
                return DDI_SHARED_TO_NEW[y]
            return None

        else:
            raise ValueError("source must be 'ddimdl' or 'ddi'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d1, d2, y = self.samples[idx]
        return (
            self.smiles_to_graph[d1],
            self.smiles_to_graph[d2],
            self.smiles_to_idx[d1],
            self.smiles_to_idx[d2],
            y
        )

def collate_fn(batch):
    g1_list, g2_list, idx1_list, idx2_list, y_list = zip(*batch)

    g1_batch = Batch.from_data_list(g1_list).to(DEVICE)
    g2_batch = Batch.from_data_list(g2_list).to(DEVICE)

    idx1 = torch.tensor(idx1_list, dtype=torch.long, device=DEVICE)
    idx2 = torch.tensor(idx2_list, dtype=torch.long, device=DEVICE)
    y = torch.tensor(y_list, dtype=torch.long, device=DEVICE)

    return g1_batch, g2_batch, idx1, idx2, y

# =========================
# EVAL
# =========================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for g1, g2, idx1, idx2, y in loader:
            logits, _, _, _ = model(g1, g2, idx1, idx2)
            loss = F.cross_entropy(logits, y)

            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            batch_count += 1

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    if len(y_true) == 0:
        return {
            "loss": None,
            "macro_f1": None,
            "weighted_f1": None,
            "macro_auc": np.nan,
            "weighted_auc": np.nan,
            "y_true": [],
            "y_pred": [],
            "y_score": []
        }

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    try:
        macro_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
        weighted_auc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
    except Exception:
        macro_auc = np.nan
        weighted_auc = np.nan

    return {
        "loss": total_loss / max(batch_count, 1),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score
    }

# =========================
# MAIN
# =========================
def main():
    start_total = time.time()

    logger.info("Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    logger.info(f"Raw DDIMDL train size: {len(train_df)}")
    logger.info(f"Raw DDIMDL val size  : {len(val_df)}")
    logger.info(f"Raw DDI test size    : {len(test_df)}")

    # save mapping
    mapping_rows = []
    for ddimdl_label, ddi_label in DDIMDL_TO_DDI.items():
        mapping_rows.append({
            "new_label": DDIMDL_SHARED_TO_NEW[ddimdl_label],
            "ddimdl_label": ddimdl_label,
            "ddi_label": ddi_label
        })
    pd.DataFrame(mapping_rows).sort_values("new_label").to_csv(CLASSMAP_CSV, index=False)
    logger.info(f"Saved class mapping to: {CLASSMAP_CSV}")

    logger.info("Loading DRKG embeddings...")
    ent_df = pd.read_csv(DRKG_ENTITIES, sep="\t", header=None, names=["entity", "eid"])
    ent2id = dict(zip(ent_df["entity"], ent_df["eid"]))
    drkg_emb_raw = np.load(DRKG_EMB)

    logger.info("Building molecular graphs...")
    all_smiles = pd.concat([
        train_df["Drug1"], train_df["Drug2"],
        val_df["Drug1"], val_df["Drug2"],
        test_df["Drug1"], test_df["Drug2"]
    ]).unique()

    smiles_to_graph = {}
    valid_smiles = []

    for s in tqdm(all_smiles, desc="Graphs"):
        g = mol_to_graph(s)
        if g is not None:
            smiles_to_graph[s] = g
            valid_smiles.append(s)

    sample_graph = next(iter(smiles_to_graph.values()))
    node_dim = sample_graph.x.shape[1]

    logger.info("Preparing DRKG matrix...")
    smiles_to_dbid = {}
    for df in [train_df, val_df, test_df]:
        for _, r in df.iterrows():
            smiles_to_dbid[r["Drug1"]] = r["Drug1_ID"]
            smiles_to_dbid[r["Drug2"]] = r["Drug2_ID"]

    smiles_to_idx = {s: i for i, s in enumerate(valid_smiles)}
    drkg_matrix = torch.zeros((len(valid_smiles), DRKG_DIM), dtype=torch.float32)

    for s in valid_smiles:
        dbid = smiles_to_dbid.get(s)
        kg_key = f"Compound::{dbid}"
        if kg_key in ent2id:
            drkg_matrix[smiles_to_idx[s]] = torch.tensor(
                drkg_emb_raw[ent2id[kg_key]],
                dtype=torch.float32
            )

    drkg_matrix = drkg_matrix.to(DEVICE)

    # datasets
    train_ds = DDIDataset(train_df, smiles_to_graph, smiles_to_idx, source="ddimdl")
    val_ds   = DDIDataset(val_df, smiles_to_graph, smiles_to_idx, source="ddimdl")
    test_ds  = DDIDataset(test_df, smiles_to_graph, smiles_to_idx, source="ddi")

    logger.info(f"Filtered train size (shared classes): {len(train_ds)}")
    logger.info(f"Filtered val size   (shared classes): {len(val_ds)}")
    logger.info(f"Filtered test size  (shared classes): {len(test_ds)}")
    logger.info(f"Number of shared classes: {NUM_SHARED_CLASSES}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # class weights from DDIMDL training labels after remapping
    train_y = np.array([y for _, _, y in train_ds.samples], dtype=np.int64)
    seen_classes = np.unique(train_y)

    cw_seen = compute_class_weight("balanced", classes=seen_classes, y=train_y)
    class_weights_np = np.ones(NUM_SHARED_CLASSES, dtype=np.float32)
    for c, w in zip(seen_classes, cw_seen):
        class_weights_np[c] = w

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

    logger.info("Building model...")
    model = MambaDDIModel(node_dim, drkg_matrix, NUM_SHARED_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_macro_f1 = -1
    patience_counter = 0

    logger.info("Starting training...")
    for ep in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")

        for g1, g2, idx1, idx2, y in pbar:
            optimizer.zero_grad()

            logits, logits_g, logits_r, gate = model(g1, g2, idx1, idx2)

            main_loss = F.cross_entropy(logits, y)
            general_loss = F.cross_entropy(logits_g, y)
            rare_loss = F.cross_entropy(logits_r, y, weight=class_weights)

            loss = main_loss + LAMBDA_GENERAL * general_loss + LAMBDA_RARE * rare_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(Loss=f"{loss.item():.4f}")

        val_metrics = evaluate(model, val_loader)

        logger.info(
            f"Epoch {ep+1}: "
            f"TrainLoss={running_loss/len(train_loader):.4f} | "
            f"ValLoss={val_metrics['loss']:.4f} | "
            f"Val Macro-F1={val_metrics['macro_f1']:.4f} | "
            f"Val Weighted-F1={val_metrics['weighted_f1']:.4f} | "
            f"Val Macro-AUC={val_metrics['macro_auc']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            torch.save(copy.deepcopy(model.state_dict()), BEST_MODEL_PT)
            patience_counter = 0
            logger.info(f"Best model saved at epoch {ep+1}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info("Loading best model...")
    model.load_state_dict(torch.load(BEST_MODEL_PT, map_location=DEVICE))

    logger.info("Evaluating on DDI test set...")
    test_metrics = evaluate(model, test_loader)

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        digits=4,
        zero_division=0
    )

    print("\n========== EXPERIMENT C (STRICT TRANSFER) ==========")
    print("Train: DDIMDL (strict common classes only)")
    print("Val  : DDIMDL held-out split")
    print("Test : DDI (strict semantic overlap only)\n")
    print(f"Shared classes: {NUM_SHARED_CLASSES}\n")
    print(report)
    print(f"Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Macro AUC: {test_metrics['macro_auc']:.4f}")
    print(f"Weighted AUC: {test_metrics['weighted_auc']:.4f}")

    with open(SUMMARY_TXT, "w") as f:
        f.write("========== EXPERIMENT C (STRICT TRANSFER) ==========\n")
        f.write("Train: DDIMDL (strict common classes only)\n")
        f.write("Val  : DDIMDL held-out split\n")
        f.write("Test : DDI (strict semantic overlap only)\n\n")
        f.write(f"Shared classes: {NUM_SHARED_CLASSES}\n\n")
        f.write(report)
        f.write(f"\nMacro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro AUC: {test_metrics['macro_auc']:.4f}\n")
        f.write(f"Weighted AUC: {test_metrics['weighted_auc']:.4f}\n")

    logger.info(f"Saved summary to: {SUMMARY_TXT}")
    logger.info(f"Done in {time.time() - start_total:.1f}s")

if __name__ == "__main__":
    main()