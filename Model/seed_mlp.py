import os, json, random, argparse, csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt

# ----------------------------
# Repro
# ----------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ----------------------------
# Dataset
# ----------------------------
def _load_vec_file(path: str) -> np.ndarray:
    """
    Robustly load a 1D float vector from a text file:
    - supports comma-separated or whitespace-separated values
    - tolerates optional [brackets]
    - also supports .npy directly
    Returns a 1D float32 numpy array.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        v = np.load(path).astype(np.float32, copy=False).squeeze()
        return v

    # Text: read once and parse
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        s = f.read()

    # Strip brackets/quotes if present
    s = s.replace("[", " ").replace("]", " ").replace("\"", " ").replace("'", " ")
    # Normalize newlines to spaces
    s = s.replace("\r", " ").replace("\n", " ")

    # Choose separator automatically
    if "," in s:
        v = np.fromstring(s, sep=",", dtype=np.float32)
    else:
        v = np.fromstring(s, sep=" ", dtype=np.float32)

    if v.ndim != 1:
        v = v.squeeze()
    if v.size == 0:
        raise ValueError(f"Failed to parse numeric vector from file: {path}")
    return v.astype(np.float32, copy=False)
class WindowDataset(Dataset):
    """CSV with columns: path,label"""
    def __init__(self, csv_path: str, input_dim: int | None = None, verify_dim: bool = True):
        self.df = pd.read_csv(csv_path)
        assert {'path','label'}.issubset(self.df.columns), "CSV must have 'path' and 'label'"
        self.df['label'] = self.df['label'].astype(int)
        self.input_dim = input_dim

        # Auto-detect dimension from first row (or verify)
        if verify_dim and len(self.df) > 0:
            v = _load_vec_file(self.df.iloc[0]['path'])
            dim = int(v.size)
            if self.input_dim is None:
                self.input_dim = dim
            else:
                assert self.input_dim == dim, f"Embedding dim {dim} != expected {self.input_dim}"

        self.n_pos = int((self.df['label']==1).sum())
        self.n_neg = int((self.df['label']==0).sum())

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v = _load_vec_file(r['path'])
        x = torch.from_numpy(v).float()          # [D]
        y = torch.tensor(r['label'], dtype=torch.float32)
        return {'x': x, 'y': y}

def collate(batch):
    X = torch.stack([b['x'] for b in batch], 0)  # [B,D]
    y = torch.stack([b['y'] for b in batch], 0)  # [B]
    return {'X': X, 'y': y}

# ----------------------------
# Model: shallow MLP
# ----------------------------
class SeedMLP(nn.Module):
    def __init__(self, in_dim=1024, hid=256, p_drop=0.2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(p_drop))
        self.head = nn.Linear(hid, 1)  # logit
    def forward(self, X):              # [B, in_dim]
        h = self.enc(X)
        return self.head(h).squeeze(-1)  # [B]

# ----------------------------
# Metrics & thresholding
# ----------------------------
def compute_pr(y_true: np.ndarray, y_score: np.ndarray, thr: float | None = None):
    ap = float(average_precision_score(y_true, y_score)) if (y_true.size and y_score.size) else 0.0
    if thr is None:
        p, r, t = precision_recall_curve(y_true, y_score)
        if t.size == 0:
            return {'ap': ap, 'precision': float(p[-1]), 'recall': float(r[-1]), 'thr': 0.5}
        f1 = (2*p*r) / (p + r + 1e-9)
        j = int(np.argmax(f1))
        thr = float(t[j]) if j < len(t) else 0.5
        return {'ap': ap, 'precision': float(p[j]), 'recall': float(r[j]), 'thr': thr}
    else:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred==1)&(y_true==1)).sum())
        fp = int(((pred==1)&(y_true==0)).sum())
        fn = int(((pred==0)&(y_true==1)).sum())
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        return {'ap': ap, 'precision': float(prec), 'recall': float(rec), 'thr': float(thr)}

def pick_threshold_cap_fp_per_hour(scores: np.ndarray, labels: np.ndarray,
                                   window_sec: float = 3.0, max_fp_per_hour: float = 1.0) -> float:
    if (labels == 0).sum() == 0:
        return 0.5
    neg = scores[labels == 0]
    neg_hours = (neg.size * window_sec) / 3600.0
    if neg_hours <= 0:
        return 0.5
    ths = np.unique(neg)
    ths = np.sort(ths)[::-1]  # high→low
    best = ths[-1] if ths.size else 0.5
    for thr in ths:
        fp = int((neg >= thr).sum())
        if (fp / neg_hours) <= max_fp_per_hour:
            best = thr; break
    return float(best)

# ----------------------------
# Confusion matrix logging helpers
# ----------------------------
def _make_confmat_figure(cm: np.ndarray, classes=("Non-WGP","WGP"), title="Confusion Matrix"):
    """Return a matplotlib Figure rendering the confusion matrix."""
    fig, ax = plt.subplots(figsize=(3.2, 3.0), dpi=120)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1], labels=classes)
    ax.set_yticks([0,1], labels=classes)
    # annotate cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=10)
    fig.tight_layout()
    return fig

def _log_confusion(writer: SummaryWriter, tag_prefix: str, y_true: np.ndarray, y_score: np.ndarray, thr: float, step: int):
    """Compute confusion matrix at threshold, log as scalars + image."""
    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0,1])
    # cm layout with labels=[0,1]: rows=true, cols=pred
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

    # Scalars
    writer.add_scalar(f"{tag_prefix}/TP", tp, step)
    writer.add_scalar(f"{tag_prefix}/FP", fp, step)
    writer.add_scalar(f"{tag_prefix}/TN", tn, step)
    writer.add_scalar(f"{tag_prefix}/FN", fn, step)

    # Image (figure)
    fig = _make_confmat_figure(cm, classes=("Non-WGP","WGP"), title=f"{tag_prefix} @ thr={thr:.3f}")
    writer.add_figure(f"{tag_prefix}/matrix", fig, global_step=step, close=True)

# ----------------------------
# Sampler helpers
# ----------------------------
def _parse_ratio(text: str) -> tuple[int, int]:
    """Parse '1:4' -> (1,4). Ensures both parts are positive ints."""
    parts = text.strip().split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("train_ratio must look like '1:4'")
    pos_k, neg_k = map(int, parts)
    if pos_k <= 0 or neg_k <= 0:
        raise argparse.ArgumentTypeError("train_ratio numbers must be positive")
    return (pos_k, neg_k)

def _class_ratio_weights(labels: np.ndarray, desired_ratio=(1, 4)) -> np.ndarray:
    """
    Per-sample weights so that expected draws match desired pos:neg ratio.
    Use with WeightedRandomSampler(replacement=True).
    """
    labels = labels.astype(int)
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    pos_k, neg_k = desired_ratio
    w_pos = float(pos_k) / n_pos
    w_neg = float(neg_k) / n_neg
    return np.where(labels == 1, w_pos, w_neg).astype(np.float64)

# ----------------------------
# Train / Eval helpers
# ----------------------------
def make_loader(csv_path: str, batch_size: int, num_workers: int,
                use_sampler: bool, device: str, desired_ratio=(1, 4)):
    ds = WindowDataset(csv_path)
    if use_sampler and ds.n_pos > 0 and ds.n_neg > 0 and len(ds) >= batch_size:
        labels = ds.df['label'].values.astype(int)
        weights = _class_ratio_weights(labels, desired_ratio=desired_ratio)
        sampler = WeightedRandomSampler(
            torch.tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True
        )
        loader = DataLoader(
            ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, collate_fn=collate, pin_memory=(device!='cpu')
        )
    else:
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, collate_fn=collate, pin_memory=(device!='cpu')
        )
    return ds, loader

def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for batch in loader:
            X = batch['X'].to(device)
            y = batch['y']
            s = torch.sigmoid(model(X)).cpu().numpy()
            y_true.append(y.numpy()); y_score.append(s)
    if len(y_true) == 0:
        return np.array([]), np.array([])
    return np.concatenate(y_true), np.concatenate(y_score)

# ----------------------------
# Train loop
# ----------------------------
def train(args):
    set_seed(42)
    device = 'cuda' if (torch.cuda.is_available() and not args.cpu) else 'cpu'

    # Loaders: TRAIN uses sampler (target ratio), VALID is natural
    train_ratio = args.train_ratio
    train_ds, train_loader = make_loader(args.train, args.batch_size, args.workers,
                                         use_sampler=True, device=device, desired_ratio=train_ratio)
    valid_ds, valid_loader = make_loader(args.valid, args.batch_size, args.workers,
                                         use_sampler=False, device=device)

    in_dim = train_ds.input_dim or 1024
    model  = SeedMLP(in_dim=in_dim, hid=args.hid, p_drop=args.dropout).to(device)

    # Loss weighting on TRAIN split
    pos_weight = None
    if train_ds.n_pos > 0 and train_ds.n_neg > 0:
        pos_weight = max(train_ds.n_neg / max(train_ds.n_pos, 1), 1.0)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device)) if pos_weight else nn.BCEWithLogitsLoss()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Logging setup ----
    writer = SummaryWriter(log_dir=args.logdir)
    csv_path = args.csv_log if args.csv_log else os.path.splitext(args.save)[0] + "_trainlog.csv"
    first_write = not os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if first_write:
        csv_writer.writerow(["epoch","train_loss","valid_AP","prec_at_F1thr","rec_at_F1thr","F1_threshold"])

    best_ap, best_epoch = -1.0, 0
    bad_epochs = 0

    try:
        for epoch in range(1, args.epochs+1):
            model.train()
            total_loss, n = 0.0, 0
            for batch in train_loader:
                X = batch['X'].to(device); y = batch['y'].to(device)
                logits = model(X)
                loss = criterion(logits, y)
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                total_loss += loss.item() * y.size(0); n += y.size(0)

            # Validation (natural)
            y_true, y_score = evaluate(model, valid_loader, device)
            pr = compute_pr(y_true, y_score) if y_true.size else {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'thr': 0.5}

            # Prints
            print(f"Epoch {epoch:02d} | TrainLoss {total_loss/max(n,1):.4f} | "
                  f"Valid AP {pr['ap']:.4f} | P {pr['precision']:.3f} R {pr['recall']:.3f} @thr≈{pr['thr']:.3f}")

            # TensorBoard scalars
            writer.add_scalar("train/loss", total_loss/max(n,1), epoch)
            writer.add_scalar("valid/AP",   pr['ap'], epoch)
            writer.add_scalar("valid/precision_at_F1thr", pr['precision'], epoch)
            writer.add_scalar("valid/recall_at_F1thr",    pr['recall'], epoch)
            writer.add_scalar("valid/F1_threshold",       pr['thr'], epoch)

            # CSV row
            csv_writer.writerow([epoch, total_loss/max(n,1), pr['ap'], pr['precision'], pr['recall'], pr['thr']])
            csv_file.flush()

            # Checkpoint by AP + early stop
            if pr['ap'] > best_ap + 1e-6:
                best_ap, best_epoch = pr['ap'], epoch
                torch.save({'model_state': model.state_dict(), 'in_dim': in_dim}, args.save)
                print(f"  ↳ Saved best to {args.save}")
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"Early stopping at epoch {epoch}. Best AP {best_ap:.4f} (epoch {best_epoch}).")
                    break

    finally:
        # Ensure file handles are closed even if interrupted
        csv_file.close()
        writer.flush(); writer.close()

    # Load best & tune threshold on validation
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt['model_state']); model.to(device).eval()

    y_true, y_score = evaluate(model, valid_loader, device)
    if y_true.size:
        if args.fp_per_hour is not None:
            thr = pick_threshold_cap_fp_per_hour(y_score, y_true, window_sec=3.0, max_fp_per_hour=args.fp_per_hour)
        else:
            thr = compute_pr(y_true, y_score)['thr']
        final = compute_pr(y_true, y_score, thr=thr)
    else:
        thr = 0.5
        final = {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'thr': thr}

    # Save meta
    meta = {
        'in_dim': int(in_dim),
        'best_val_ap': float(best_ap),
        'trained_epochs': int(best_epoch),
        'pos_in_train': int(train_ds.n_pos),
        'neg_in_train': int(train_ds.n_neg),
        'pos_in_valid': int(valid_ds.n_pos),
        'neg_in_valid': int(valid_ds.n_neg),
        'threshold': float(final['thr']),
        'precision_at_thr': float(final['precision']),
        'recall_at_thr': float(final['recall']),
        'fp_per_hour_target': args.fp_per_hour if args.fp_per_hour is not None else None,
        'model_path': args.save,
    }
    with open(os.path.splitext(args.save)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Tuned thr {final['thr']:.4f} → P={final['precision']:.3f}, R={final['recall']:.3f}, AP={final['ap']:.4f}")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="training.csv", help="path to training.csv")
    ap.add_argument("--valid", default="valid.csv", help="path to valid.csv")
    ap.add_argument("--save",  default="seed_mlp.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=4, help="early stop patience")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--hid", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--fp_per_hour", type=float, default=1.0, help="cap FP/h on validation; set None to disable")
    ap.add_argument("--train_ratio", type=_parse_ratio, default="1:4",
                    help="target pos:neg ratio for TRAIN sampler, e.g., '1:3', '1:4', '1:5'")
    ap.add_argument("--logdir", default="runs/seed_mlp", help="TensorBoard log directory")
    ap.add_argument("--csv_log", default=None, help="Optional path for CSV log file; defaults to <save>_trainlog.csv")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()
    train(args)


### run code
# cd D:/PhD/WGP_model/model_dummy/train
# python seed_mlp.py --train training.csv --valid valid.csv
