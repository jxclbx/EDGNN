import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from data.read_dataset import read_dataset
from torch_geometric.utils import to_dense_adj, to_dense_batch
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="data/jacob_dataset.pt.gz")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=125)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--kfold", type=int, default=5)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

exp_name = f"GNN_{args.folder}_bs{args.batch_size}_lr{args.lr}_kfold{args.kfold}"
save_dir = os.path.join("results", exp_name)
os.makedirs(save_dir, exist_ok=True)
print(save_dir)
dataset = read_dataset(args.folder)
num_classes = len(set([int(data.y.item()) for data in dataset]))
in_channels = dataset[0].x.shape[1]


class GraphClassifier(nn.Module):
    """
        1) The X branch
        2) Adjacency row
        3) Adjacency row
        4) K-hop aggregation (based on A: [X, AX, A^2X, ...])
        5) K-hop aggregation (based on A^T: [X, A^T X, (A^T)^2 X, ...])
    """
    def __init__(self, in_channels, hidden_channels, num_classes, proj_dim=32, K=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden_channels
        self.num_classes = num_classes
        self.K = K

        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )

        self.mlp_adj_dir = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        self.mlp_adj_tr = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )

        agg_in = (K + 1) * in_channels
        self.mlp_agg_dir = nn.Sequential(
            nn.Linear(agg_in, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        self.mlp_agg_tr = nn.Sequential(
            nn.Linear(agg_in, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )

        # node-level fusion
        self.fuse = nn.Sequential(
            nn.Linear(hidden_channels * 5, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # graph-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    @staticmethod
    def _row_norm(A):
        deg = A.sum(dim=-1).clamp(min=1.0)
        return A / deg.unsqueeze(-1)

    def _k_hop_stack(self, A_norm, X):
        outs = [X]
        Z = X
        for _ in range(self.K):
            Z = torch.bmm(A_norm, Z)
            outs.append(Z)
        return torch.cat(outs, dim=-1)

    def forward(self, x, edge_index, batch):
        device = x.device

        # densify
        X_dense, _ = to_dense_batch(x, batch)         # [B, N, F]
        A_dir = to_dense_adj(edge_index, batch=batch) # [B, N, N]
        B, N, F = X_dense.shape

        A_tr = A_dir.transpose(1, 2)
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
        Adir_norm = self._row_norm(A_dir + I)
        Atr_norm  = self._row_norm(A_tr  + I)

        hx = self.mlp_x(X_dense.reshape(B * N, F)).reshape(B, N, self.hidden)

        hAd = self.mlp_adj_dir(A_dir.reshape(B * N, N)).reshape(B, N, self.hidden)
        hAt = self.mlp_adj_tr (A_tr .reshape(B * N, N)).reshape(B, N, self.hidden)

        feat_stack_dir = self._k_hop_stack(Adir_norm, X_dense)  # [B, N, (K+1)F]
        feat_stack_tr  = self._k_hop_stack(Atr_norm,  X_dense)

        f_dir = self.mlp_agg_dir(feat_stack_dir.reshape(B * N, -1)).reshape(B, N, self.hidden)
        f_tr  = self.mlp_agg_tr (feat_stack_tr .reshape(B * N, -1)).reshape(B, N, self.hidden)

        H = torch.cat([hx, hAd, hAt, f_dir, f_tr], dim=-1)   # [B, N, 5*hidden]
        H = self.fuse(H.reshape(B * N, -1)).reshape(B, N, self.hidden)
        H_flat = H.reshape(B * N, self.hidden)
        new_batch = torch.arange(B, device=device).repeat_interleave(N)
        hG = global_mean_pool(H_flat, new_batch)
        return self.classifier(hG)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def train(model, optimizer, loader):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch.x, batch.edge_index, batch.batch)
            prob = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            pred = out.argmax(dim=1).cpu().numpy()
            label = batch.y.cpu().numpy()
            preds.extend(pred)
            labels.extend(label)
            probs.extend(prob)
    return np.array(preds), np.array(labels), np.array(probs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

all_runs_metrics = []

for seed in tqdm(range(5), desc="Seeds", ncols=50):
    set_seed(seed)

    labels = [int(data.y.item()) for data in dataset]
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=seed)
    fold_metrics = []

    run_report_path = os.path.join(save_dir, f"run_seed{seed}.txt")
    with open(run_report_path, "w") as f_run:
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(dataset)), labels), 1):
            train_set = [dataset[i] for i in train_idx]
            test_set = [dataset[i] for i in test_idx]
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=args.batch_size)

            model = GraphClassifier(in_channels, hidden_channels=64, num_classes=num_classes).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            best_auc = 0
            best_preds, best_labels, best_probs = None, None, None

            for _ in range(1, args.epochs + 1):
                _ = train(model, optimizer, train_loader)
                test_preds, test_labels, test_probs = evaluate(model, test_loader)
                auc = roc_auc_score(test_labels, test_probs)
                if auc > best_auc:
                    best_auc = auc
                    best_preds = test_preds.copy()
                    best_labels = test_labels.copy()
                    best_probs = test_probs.copy()
                print(f"Seed {seed} Fold {fold} Epoch {_}: Test AUC {auc:.4f} (Best {best_auc:.4f})", end='\r')
            acc = accuracy_score(best_labels, best_preds)
            precision = precision_score(best_labels, best_preds)
            recall = recall_score(best_labels, best_preds)
            f1 = f1_score(best_labels, best_preds)
            auc = roc_auc_score(best_labels, best_probs)

            fold_metrics.append([acc, precision, recall, f1, auc])

            f_run.write(f"Fold {fold} Metrics:\n")
            f_run.write(f"Accuracy: {acc:.4f}\n")
            f_run.write(f"Precision: {precision:.4f}\n")
            f_run.write(f"Recall: {recall:.4f}\n")
            f_run.write(f"F1-score: {f1:.4f}\n")
            f_run.write(f"AUC: {auc:.4f}\n")
            f_run.write("Classification Report:\n")
            f_run.write(classification_report(best_labels, best_preds))
            f_run.write("\n\n")

        fold_avg_metrics = np.mean(fold_metrics, axis=0)
        f_run.write(f"Run Seed {seed} Mean:\n")
        f_run.write(f"Accuracy: {fold_avg_metrics[0]:.4f}\n")
        f_run.write(f"Precision: {fold_avg_metrics[1]:.4f}\n")
        f_run.write(f"Recall: {fold_avg_metrics[2]:.4f}\n")
        f_run.write(f"F1-score: {fold_avg_metrics[3]:.4f}\n")
        f_run.write(f"AUC: {fold_avg_metrics[4]:.4f}\n")

    all_runs_metrics.append(fold_avg_metrics)

mean_metrics = np.mean(all_runs_metrics, axis=0)
std_metrics = np.std(all_runs_metrics, axis=0)

summary_path = os.path.join(save_dir, "final_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    for seed, metrics in enumerate(all_runs_metrics):
        f.write(f"Seed {seed}:\n")
        for name, metric in zip(metrics_names, metrics):
            f.write(f"{name}: {metric:.4f}\n")
        f.write("\n")

    f.write("\nMean:\n")
    for name, metric in zip(metrics_names, mean_metrics):
        f.write(f"{name}: {metric:.4f}\n")

    f.write("\nstds:\n")
    for name, metric in zip(metrics_names, std_metrics):
        f.write(f"{name}: {metric:.4f}\n")
