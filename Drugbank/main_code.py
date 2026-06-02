#!/usr/bin/env python3
"""
GCN + DRKG + Mamba2 (DDI) WITH ATTENTION + Rare-Class Ensemble
CSV FORMAT: Drug1_ID,Drug1,Drug2_ID,Drug2,Y
"""

# =========================
# IMPORTS
# =========================
import os
import time
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, TensorDataset
from mamba_ssm import Mamba2
from sklearn.metrics import classification_report
from collections import Counter

# =========================
# CONFIG
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE = "/home2/jessygrace.polinati/Abhi/dd"
TRAIN_CSV = os.path.join(BASE, "train_ddi.csv")
VAL_CSV   = os.path.join(BASE, "val_ddi.csv")
TEST_CSV  = os.path.join(BASE, "test_ddi.csv")
SUMMARY_TXT = os.path.join(BASE, "v3.4_drkg_ensemble.txt")

# DRKG
DRKG_ENTITIES = os.path.join(BASE, "drkg/embed/entities.tsv")
DRKG_EMB      = os.path.join(BASE, "drkg/embed/DRKG_TransE_l2_entity.npy")

# GNN
GNN_HIDDEN = 128
GNN_OUT = 64
DRKG_DIM = 400
FUSED_DIM = GNN_OUT + DRKG_DIM

# Training
BATCH_SIZE = 512
LR = 1e-4
EPOCHS = 20

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("GCN_DRKG_MAMBA_ENSEMBLE")

# =========================
# ATOM FEATURES / GRAPH FUNCTIONS (same as original)
# =========================
def atom_features(atom):
    atom_type = [0]*64; atom_type[min(atom.GetAtomicNum(),63)] = 1
    chiral = [0]*7; chiral[min(int(atom.GetChiralTag()),6)] = 1
    degree = [0]*11; degree[min(atom.GetDegree(),10)] = 1
    fc = [0]*12; idx = atom.GetFormalCharge()+6; 
    if 0<=idx<12: fc[idx]=1
    nH=[0]*5; nH[min(atom.GetTotalNumHs(),4)]=1
    hybrid=[0]*7; hybrid[min(int(atom.GetHybridization()),6)]=1
    aromatic=[int(atom.GetIsAromatic())]; ring=[int(atom.IsInRing())]
    return torch.tensor(atom_type+chiral+degree+fc+nH+hybrid+aromatic+ring,dtype=torch.float32)

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    mol = Chem.RemoveHs(mol)
    atoms = mol.GetAtoms()
    if len(atoms)==0: return None
    x = torch.stack([atom_features(a) for a in atoms])
    edges=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        edges+=[[i,j],[j,i]]
    edge_index = torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

class AttentivePooling(nn.Module):
    def __init__(self,dim): super().__init__(); self.attn=nn.Linear(dim,1)
    def forward(self,x,batch):
        scores=torch.exp(self.attn(x))
        denom=torch.zeros(batch.max()+1,1,device=x.device).index_add_(0,batch,scores)
        alpha = scores/(denom[batch]+1e-9)
        return torch.zeros(batch.max()+1,x.size(1),device=x.device).index_add_(0,batch,alpha*x)

class SimpleGCN(nn.Module):
    def __init__(self,in_dim,hidden,out_dim):
        super().__init__()
        self.conv1=GCNConv(in_dim,hidden)
        self.conv2=GCNConv(hidden,out_dim)
        self.pool=AttentivePooling(out_dim)
    def forward(self,x,edge_index,batch):
        x=F.relu(self.conv1(x,edge_index))
        x=self.conv2(x,edge_index)
        return self.pool(x,batch)

class DrugPairAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.q=nn.Linear(dim,dim)
        self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim)
        self.scale = dim**-0.5
    def forward(self,x):
        Q=self.q(x); K=self.k(x); V=self.v(x)
        attn=torch.matmul(Q,K.transpose(-2,-1))*self.scale
        attn=F.softmax(attn,dim=-1)
        return torch.matmul(attn,V)

class MambaClassifier(nn.Module):
    def __init__(self,input_dim,n_class,mamba_dim=512):
        super().__init__()
        self.proj=nn.Linear(input_dim,mamba_dim)
        self.attn=DrugPairAttention(mamba_dim)
        self.mamba=Mamba2(d_model=mamba_dim,d_state=16,d_conv=4,expand=2)
        self.norm=nn.LayerNorm(mamba_dim)
        self.fc=nn.Linear(mamba_dim,n_class)
    def forward(self,x):
        x=self.proj(x)
        x=self.attn(x)+x
        x=self.mamba(x)
        x=self.norm(x)
        x=x.mean(dim=1)
        return self.fc(x)

# =========================
# ENSEMBLE CLASS
# =========================
class EnsembleClassifier(nn.Module):
    def __init__(self,input_dim,n_class):
        super().__init__()
        # Two classifiers
        self.model_dominant = MambaClassifier(input_dim,n_class)
        self.model_rare = MambaClassifier(input_dim,n_class)
    def forward(self,x):
        logits_dom = self.model_dominant(x)
        logits_rare = self.model_rare(x)
        # Weighted sum: boost rare-class logits
        weights = torch.ones_like(logits_dom)
        # Example: boost classes 0 and 1 if rare
        rare_classes = [0,1]  # adjust based on your dataset
        for c in rare_classes:
            weights[:,c] = 2.0
        combined = logits_dom + logits_rare * weights
        return combined

# =========================
# MAIN
# =========================
def main():
    start=time.time()
    logger.info("Device: %s",DEVICE)
    # Load CSVs
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # DRKG embeddings
    logger.info("Loading DRKG embeddings...")
    ent_df = pd.read_csv(DRKG_ENTITIES, sep="\t", header=None, names=["entity","eid"])
    ent2id = dict(zip(ent_df["entity"],ent_df["eid"]))
    drkg_emb = np.load(DRKG_EMB)
    def get_kg_emb(dbid):
        key=f"Compound::{dbid}"
        return drkg_emb[ent2id[key]] if key in ent2id else None

    # Build graphs
    all_smiles=pd.concat([train_df["Drug1"],train_df["Drug2"],val_df["Drug1"],val_df["Drug2"],test_df["Drug1"],test_df["Drug2"]]).unique()
    smiles_to_graph={s:mol_to_graph(s) for s in tqdm(all_smiles)}
    smiles_to_graph={k:v for k,v in smiles_to_graph.items() if v is not None}

    node_dim = next(iter(smiles_to_graph.values())).num_node_features
    gcn = SimpleGCN(node_dim,GNN_HIDDEN,GNN_OUT).to(DEVICE)
    gcn.eval()

    smiles_to_dbid={}
    for df in [train_df,val_df,test_df]:
        for _,r in df.iterrows():
            smiles_to_dbid[r["Drug1"]]=r["Drug1_ID"]
            smiles_to_dbid[r["Drug2"]]=r["Drug2_ID"]

    # Compute embeddings
    drug_emb={}
    with torch.no_grad():
        for smi,g in tqdm(smiles_to_graph.items()):
            if smi not in smiles_to_dbid: continue
            kg=get_kg_emb(smiles_to_dbid[smi])
            if kg is None: continue
            g=g.to(DEVICE)
            batch=torch.zeros(g.num_nodes,dtype=torch.long,device=DEVICE)
            gcn_e=gcn(g.x,g.edge_index,batch).squeeze(0).cpu().numpy()
            drug_emb[smi]=np.concatenate([gcn_e,kg],axis=0)

    def build_xy(df):
        X,y=[],[]
        for _,r in df.iterrows():
            if r["Drug1"] in drug_emb and r["Drug2"] in drug_emb:
                X.append(np.stack([drug_emb[r["Drug1"]],drug_emb[r["Drug2"]]]))
                y.append(int(r["Y"]))
        return np.array(X,np.float32), np.array(y,np.int64)

    X_train,y_train=build_xy(train_df)
    X_val,y_val=build_xy(val_df)
    X_test,y_test=build_xy(test_df)

    train_loader=DataLoader(TensorDataset(torch.tensor(X_train),torch.tensor(y_train)),batch_size=BATCH_SIZE,shuffle=True)
    val_loader=DataLoader(TensorDataset(torch.tensor(X_val),torch.tensor(y_val)),batch_size=BATCH_SIZE)
    test_loader=DataLoader(TensorDataset(torch.tensor(X_test),torch.tensor(y_test)),batch_size=BATCH_SIZE)

    # Ensemble model
    model = EnsembleClassifier(FUSED_DIM,len(np.unique(y_train))).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    criterion = nn.CrossEntropyLoss()

    # Count class frequency to guide rare-class weighting
    counts = Counter(y_train)
    rare_classes = [c for c,f in counts.items() if f<np.percentile(list(counts.values()),20)]
    logger.info("Rare classes: %s",rare_classes)

    # Training loop
    for ep in range(EPOCHS):
        model.train()
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            # Optionally apply class weights
            weights = torch.ones_like(logits)
            for c in rare_classes:
                weights[:,c] = 2.0
            loss = (F.cross_entropy(logits,yb,reduction='none')*weights.gather(1,yb.unsqueeze(1)).squeeze()).mean()
            loss.backward()
            optimizer.step()
        logger.info("Epoch %d/%d done",ep+1,EPOCHS)

    # Evaluation
    model.eval()
    yt,yp=[],[]
    with torch.no_grad():
        for xb,yb in test_loader:
            logits = model(xb.to(DEVICE))
            yp.extend(logits.argmax(1).cpu().numpy())
            yt.extend(yb.numpy())

    report = classification_report(yt,yp,zero_division=0)
    logger.info("\nTEST REPORT:\n%s",report)
    with open(SUMMARY_TXT,"w") as f:
        f.write(report)
    logger.info("Finished in %.1f sec",time.time()-start)

if __name__=="__main__":
    main()
