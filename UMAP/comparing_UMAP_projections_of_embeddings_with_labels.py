import os, glob, numpy as np, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
# Directory with embeddings (txt or npy files)
EMBED_DIR = r"C:/WGP/captive_calls/embeddings/rising_step_syllables_without_jinnung/rising_step_syllables_without_jinnung_no_whitespaces"
FILE_EXT  = "*.txt"   # change to "*.npy" if you saved embeddings in numpy format

paths = sorted(glob.glob(os.path.join(EMBED_DIR, FILE_EXT)))
if not paths:
    raise SystemExit(f"No embedding files found in: {EMBED_DIR}")

# ---------------- Load embeddings + labels ----------------
embeddings, bird_ids = [], []

for p in paths:
    # Load embedding (assume 1D vector; average if multiple rows)
    emb = np.loadtxt(p, delimiter=",")
    if emb.ndim > 1:
        emb = emb.mean(axis=0)
    embeddings.append(emb)

    # Extract bird ID from filename (before first underscore)
    fname = os.path.basename(p)
    bird_id = fname.split("_")[0]
    bird_ids.append(bird_id)

X = np.vstack(embeddings).astype(np.float32)
y = np.array(bird_ids)

print(f"Loaded {len(X)} embeddings of dimension {X.shape[1]}")
print("Unique IDs:", np.unique(y))

# ---------------- UMAP (unlabeled) + trustworthiness, silhouette ----------------
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist
from collections import Counter

# Optional: normalize rows for cosine geometry
Xn = normalize(X, norm="l2", axis=1)

### UMAP PARAMETERS ###
nearest_neighbours = 5
min_dist = 0.1
seeds = [42, 43, 44, 45, 46]
trustworthiness_nn = 2

unique_labels = np.unique(y)
counts = Counter(y)
print("Label counts:", dict(counts))

compute_silhouette = len(unique_labels) > 1

Zs, trusts, sils, sils_by_label_list = [], [], [], []

for s in seeds:
    reducer = umap.UMAP(
        n_neighbors=nearest_neighbours,
        min_dist=min_dist,
        metric="cosine",
        random_state=s
    )
    Z = reducer.fit_transform(Xn)
    Zs.append(Z)

    # Trustworthiness
    t = trustworthiness(Xn, Z, n_neighbors=trustworthiness_nn, metric="cosine")
    trusts.append(t)

    # Silhouette
    if compute_silhouette:
        try:
            sil = silhouette_score(Z, y, metric="euclidean")
            s_samples = silhouette_samples(Z, y, metric="euclidean")
            sil_by_label = {lbl: float(np.mean(s_samples[y == lbl])) for lbl in unique_labels}
        except Exception:
            sil = np.nan
            sil_by_label = {lbl: np.nan for lbl in unique_labels}
        sils.append(sil)
        sils_by_label_list.append(sil_by_label)

# Stability across seeds
ref_d = pdist(Zs[0], metric="euclidean")
stabilities = []
for Z in Zs[1:]:
    d = pdist(Z, metric="euclidean")
    r = np.corrcoef(ref_d, d)[0, 1]
    stabilities.append(r)

# ---------------- Print summary ----------------
print(f"\nTrustworthiness (k={trustworthiness_nn}, metric=cosine) over seeds {seeds}:")
for s, t in zip(seeds, trusts):
    print(f"  seed={s}: {t:.4f}")
print(f"  mean ± std: {np.mean(trusts):.4f} ± {np.std(trusts):.4f}")

if compute_silhouette:
    print(f"\nSilhouette over seeds {seeds}:")
    for i, s in enumerate(seeds):
        print(f"  seed={s}: overall={sils[i]:.4f}  per-label=" +
              ", ".join([f"{lbl}:{sils_by_label_list[i][lbl]:.3f}" for lbl in unique_labels]))
    print(f"  mean ± std (overall): {np.nanmean(sils):.4f} ± {np.nanstd(sils):.4f}")

print(f"\nEmbedding stability vs seed {seeds[0]} (pairwise-distance correlation):")
for s, r in zip(seeds[1:], stabilities):
    print(f"  seed={s}: r={r:.4f}")
print(f"  mean r: {np.mean(stabilities):.4f}")

# ---------------- Plot ----------------
Z = Zs[0]
plt.figure(figsize=(8, 8))
unique_birds = np.unique(y)
cmap = plt.cm.get_cmap("tab20", len(unique_birds))

for i, bird in enumerate(unique_birds):
    idx = np.where(y == bird)[0]
    plt.scatter(
        Z[idx, 0], Z[idx, 1],
        s=60, color=cmap(i), alpha=0.9, edgecolor="black",
        label=bird
    )

title_extra = f" | sil={sils[0]:.3f} (overall)" if compute_silhouette else ""
plt.title(
    f"UMAP (cosine) — seed {seeds[0]} | nn={nearest_neighbours}, min_dist={min_dist}\n"
    f"trust={trusts[0]:.3f} | mean trust={np.mean(trusts):.3f} | "
    f"mean stability r={np.mean(stabilities):.3f}{title_extra}"
)
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.1, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()