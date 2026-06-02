#!/usr/bin/env python3
"""
Improved Multiclass DDI Prediction with:
- Trainable GAT + residuals
- DRKG embeddings
- Bidirectional Mamba experts
- Rare-aware expert with auxiliary loss
- Confidence-aware gating
- Train on TRAIN_CSV, evaluate on TEST_CSV (no validation split)
"""

import os, time, random, logging
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
except ImportError:
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
TRAIN_CSV = "/home2/jessygrace.polinati/Abhi/dd/Generalisation/ddimdl_final_processed_1.csv"
TEST_CSV  = "/home2/jessygrace.polinati/Abhi/dd/Generalisation/drugbank_final_processed_1.csv"

DRKG_ENTITIES = "/home2/jessygrace.polinati/Abhi/dd/drkg/embed/entities.tsv"
DRKG_EMB      = "/home2/jessygrace.polinati/Abhi/dd/drkg/embed/DRKG_TransE_l2_entity.npy"

SUMMARY_TXT   = "/home2/jessygrace.polinati/Abhi/dd/Generalisation/5.4.1_ddimdl_db.txt"
BEST_MODEL_PT = "/home2/jessygrace.polinati/Abhi/dd/Generalisation/mambaddi_best_1.pt"

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
EPOCHS = 30
D_MODEL = 256
TAU = 0.65

LAMBDA_GENERAL = 0.2
LAMBDA_RARE = 0.4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("MambaDDI_Improved")


# =========================
# ATOM FEATURES
# =========================
def atom_features(atom):
    atom_type = [0]*64
    atom_type[min(atom.GetAtomicNum(),63)] = 1

    chiral = [0]*7
    chiral[min(int(atom.GetChiralTag()),6)] = 1

    degree = [0]*11
    degree[min(atom.GetDegree(),10)] = 1

    fc = [0]*12
    idx = atom.GetFormalCharge()+6
    if 0 <= idx < 12:
        fc[idx] = 1

    nH = [0]*5
    nH[min(atom.GetTotalNumHs(),4)] = 1

    hybrid = [0]*7
    hybrid[min(int(atom.GetHybridization()),6)] = 1

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
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

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
        cls = self.cls_token.expand(B,1,-1)

        seq_ab = torch.cat([cls,x],dim=1)
        out_ab = self.mamba(seq_ab)
        if isinstance(out_ab, tuple):
            out_ab = out_ab[0]
        rep_ab = out_ab[:,0]

        x_rev = torch.flip(x,dims=[1])
        seq_ba = torch.cat([cls,x_rev],dim=1)
        out_ba = self.mamba(seq_ba)
        if isinstance(out_ba, tuple):
            out_ba = out_ba[0]
        rep_ba = out_ba[:,0]

        rep = 0.5*(rep_ab+rep_ba)
        rep = self.norm(rep)

        return self.head(rep), rep


class RareExpert(nn.Module):
    def __init__(self, input_dim, n_class, d_model=D_MODEL):
        super().__init__()

        self.proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model))

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
        cls = self.cls_token.expand(B,1,-1)

        seq_ab = torch.cat([cls,x],dim=1)
        out_ab = self.mamba(seq_ab)
        if isinstance(out_ab, tuple):
            out_ab = out_ab[0]
        out_ab = self.norm1(out_ab)
        out_ab = self.transformer(out_ab)
        rep_ab = out_ab[:,0]

        x_rev = torch.flip(x,dims=[1])
        seq_ba = torch.cat([cls,x_rev],dim=1)
        out_ba = self.mamba(seq_ba)
        if isinstance(out_ba, tuple):
            out_ba = out_ba[0]
        out_ba = self.norm1(out_ba)
        out_ba = self.transformer(out_ba)
        rep_ba = out_ba[:,0]

        rep = 0.5*(rep_ab+rep_ba)
        rep = self.norm2(rep)

        return self.head(rep), rep


# =========================
# GATING
# =========================
class GatingNetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.fc2 = nn.Linear(input_dim//2, 2)

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

        return torch.cat([gnn_vec, kg_vec], dim=1)

    def forward(self, g1_batch, g2_batch, idx1, idx2):
        emb1 = self.encode_drugs(g1_batch, idx1)
        emb2 = self.encode_drugs(g2_batch, idx2)

        pair_x = torch.stack([emb1, emb2], dim=1)   # [B, 2, FUSED_DIM]

        logits_g, embed_g = self.general_expert(pair_x)
        logits_r, embed_r = self.rare_expert(pair_x)

        gate, conf_g, conf_r = self.gating(
            embed_g,
            embed_r,
            logits_g,
            logits_r
        )

        # -----------------------------------
        # SAFE TAU-AWARE RARE EXPERT BOOST
        # -----------------------------------
        low_conf_mask = (conf_g < TAU).float()

        rare_boost = torch.cat([
            torch.zeros_like(low_conf_mask),
            0.15 * low_conf_mask
        ], dim=1)

        gate_adjust = gate + rare_boost
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
    def __init__(self, df, smiles_to_graph, smiles_to_idx):
        self.samples = []

        for _, row in df.iterrows():
            d1, d2, y = row["Drug1"], row["Drug2"], int(row["Y"])

            if d1 in smiles_to_graph and d2 in smiles_to_graph:
                self.samples.append((d1,d2,y))

        self.smiles_to_graph = smiles_to_graph
        self.smiles_to_idx = smiles_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d1,d2,y = self.samples[idx]

        return (
            self.smiles_to_graph[d1],
            self.smiles_to_graph[d2],
            self.smiles_to_idx[d1],
            self.smiles_to_idx[d2],
            y
        )


def collate_fn(batch):
    g1,g2,idx1,idx2,y = zip(*batch)

    return (
        Batch.from_data_list(g1).to(DEVICE),
        Batch.from_data_list(g2).to(DEVICE),
        torch.tensor(idx1,dtype=torch.long,device=DEVICE),
        torch.tensor(idx2,dtype=torch.long,device=DEVICE),
        torch.tensor(y,dtype=torch.long,device=DEVICE)
    )


# =========================
# EVALUATION
# =========================
def evaluate(model, loader):
    model.eval()

    y_true,y_pred,y_score = [],[],[]
    total_loss = 0

    with torch.no_grad():
        for g1,g2,idx1,idx2,y in loader:
            logits,_,_,_ = model(g1,g2,idx1,idx2)

            loss = F.cross_entropy(logits,y)

            probs = F.softmax(logits,dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    return {
        "loss": total_loss/len(loader),
        "macro_f1": f1_score(y_true,y_pred,average="macro"),
        "weighted_f1": f1_score(y_true,y_pred,average="weighted"),
        "macro_auc": roc_auc_score(y_true,y_score,multi_class="ovr",average="macro"),
        "weighted_auc": roc_auc_score(y_true,y_score,multi_class="ovr",average="weighted"),
        "y_true": y_true,
        "y_pred": y_pred
    }


# =========================
# MAIN
# =========================
def main():
    logger.info("Loading CSVs...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    logger.info("Loading DRKG...")
    ent_df = pd.read_csv(DRKG_ENTITIES, sep="\t", header=None, names=["entity","eid"])
    ent2id = dict(zip(ent_df["entity"], ent_df["eid"]))
    drkg_emb_raw = np.load(DRKG_EMB)

    logger.info("Building graphs...")
    all_smiles = pd.concat([
        train_df["Drug1"], train_df["Drug2"],
        test_df["Drug1"], test_df["Drug2"]
    ]).unique()

    smiles_to_graph = {}
    valid_smiles = []

    for s in tqdm(all_smiles):
        g = mol_to_graph(s)
        if g is not None:
            smiles_to_graph[s] = g
            valid_smiles.append(s)

    node_dim = next(iter(smiles_to_graph.values())).x.shape[1]

    smiles_to_dbid = {}
    for df in [train_df,test_df]:
        for _,r in df.iterrows():
            smiles_to_dbid[r["Drug1"]] = r["Drug1_ID"]
            smiles_to_dbid[r["Drug2"]] = r["Drug2_ID"]

    smiles_to_idx = {s:i for i,s in enumerate(valid_smiles)}

    drkg_matrix = torch.zeros((len(valid_smiles),DRKG_DIM),dtype=torch.float32)

    for s in valid_smiles:
        dbid = smiles_to_dbid[s]
        kg_key = f"Compound::{dbid}"

        if kg_key in ent2id:
            drkg_matrix[smiles_to_idx[s]] = torch.tensor(
                drkg_emb_raw[ent2id[kg_key]],
                dtype=torch.float32
            )

    drkg_matrix = drkg_matrix.to(DEVICE)

    train_ds = DDIDataset(train_df, smiles_to_graph, smiles_to_idx)
    test_ds  = DDIDataset(test_df, smiles_to_graph, smiles_to_idx)

    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn)

    classes = np.unique(train_df["Y"].values.astype(np.int64))
    n_class = len(classes)

    cw = compute_class_weight(
        "balanced",
        classes=classes,
        y=train_df["Y"].values.astype(np.int64)
    )

    class_weights = torch.tensor(cw,dtype=torch.float32,device=DEVICE)

    model = MambaDDIModel(node_dim,drkg_matrix,n_class).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    logger.info("Training...")

    for ep in range(EPOCHS):
        model.train()
        running_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}")

        for g1,g2,idx1,idx2,y in pbar:
            optimizer.zero_grad()

            logits,logits_g,logits_r,_ = model(g1,g2,idx1,idx2)

            main_loss = F.cross_entropy(logits,y)
            general_loss = F.cross_entropy(logits_g,y)
            rare_loss = F.cross_entropy(logits_r,y,weight=class_weights)

            loss = main_loss + LAMBDA_GENERAL*general_loss + LAMBDA_RARE*rare_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        logger.info(f"Epoch {ep+1}: Train Loss = {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), BEST_MODEL_PT)

    logger.info("Evaluating on Test...")
    test_metrics = evaluate(model,test_loader)

    report = classification_report(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        digits=4,
        zero_division=0
    )

    print(report)
    print("Macro F1:", test_metrics["macro_f1"])
    print("Weighted F1:", test_metrics["weighted_f1"])
    print("Macro AUC:", test_metrics["macro_auc"])
    print("Weighted AUC:", test_metrics["weighted_auc"])

    with open(SUMMARY_TXT,"w") as f:
        f.write(report)
        f.write(f"\nMacro F1: {test_metrics['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {test_metrics['weighted_f1']:.4f}\n")
        f.write(f"Macro AUC: {test_metrics['macro_auc']:.4f}\n")
        f.write(f"Weighted AUC: {test_metrics['weighted_auc']:.4f}\n")


if __name__ == "__main__":
    main()