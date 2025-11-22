# Simple pipeline: no preprocessing, no log time rescale.
# Pad Mel spectrograms to the longest one; keep plot time scale normal (ms).
import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
# AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/D_syllables"  # laptop
AUDIO_DIR = "C:/WGP/captive_calls/ch.1/analyses_for_conference/dataset/syllab_nr"

paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "**", "*.wav"), recursive=True))
print(paths)
if not paths:
    raise SystemExit(f"No .wav files found in: {AUDIO_DIR}")

SR     = 22000
N_FFT  = 2048
HOP    = 128
N_MELS = 128
FMIN, FMAX = 1000, 4200

# Root compression for a touch more sensitivity (features only; still "no preprocessing")
POWER  = 0.3   # try 0.5–0.7; 1.0 = amplitude, 2.0 = power

# ---------------- Load waveforms + labels (no RMS/dB/z-score) ----------------
waveforms, bird_ids, call_ids, bird_and_call_ids, durs = [], [], [], [], []

print("File durations:")
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)
    dur_s = len(y) / sr
    print(f"  {os.path.basename(p):<40} {dur_s*1000:7.1f} ms  ({dur_s:.3f} s)")
    waveforms.append(y)
    bird_ids.append(os.path.basename(p)[:5])
    call_ids.append(os.path.basename(p)[6:-7])
    bird_and_call_ids.append(os.path.basename(p)[0:-4])
    durs.append(dur_s)

durs = np.array(durs)
print(f"\nTotal files: {len(durs)} | mean: {durs.mean()*1000:.1f} ms | "
      f"median: {np.median(durs)*1000:.1f} ms | min: {durs.min()*1000:.1f} ms | "
      f"max: {durs.max()*1000:.1f} ms\n")

# ---------------- Mel helper (no dB/z-score; avoid STFT auto-padding) ----------------
def mel_raw(y):
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=POWER, center=False  # keep time axis "true"
    )  # (n_mels, T)

# ---------------- Build Mel spectrograms (un-padded) ----------------
specs = [mel_raw(y) for y in waveforms]          # un-padded (for plotting, true time)
lengths = [S.shape[1] for S in specs]
T_max   = max(lengths)

# ---------------- Pad to the longest spectrogram (features for UMAP) ----------------
specs_padded = []
for S in specs:
    T = S.shape[1]
    if T < T_max:
        S = np.pad(S, ((0,0), (0, T_max - T)), mode="constant")
    else:
        S = S[:, :T_max]
    specs_padded.append(S)

# ---------------- Build feature matrix X for UMAP ----------------
# (N samples) x (N_MELS * T_max features)
X = np.vstack([S.ravel() for S in specs_padded]).astype(np.float32)





# ---------------- UMAP (unlabeled) + trustworthiness, stability, silhouette over seeds ----------------
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.spatial.distance import pdist
from collections import Counter
import numpy as np
import umap
import matplotlib.pyplot as plt

# Optional: for cosine geometry it's often helpful to L2-normalize each row.
# If you don't want that, just set: Xn = X
Xn = normalize(X, norm="l2", axis=1)

### UMAP PARAMETERS ###
nearest_neighbours = 5         # try {2, 3, 5}
min_dist = 0.1              # try {0.0, 0.01, 0.05, 0.1}

seeds = [42, 43, 44, 45, 46]   # use the same set each run
trustworthiness_nn = 2         # trustworthiness' nearest-neighbour parameter

# Labels for silhouette (bird IDs)
y = np.array(call_ids)                                                             #todo remember to change this if exploring different label variable!!
unique_labels = np.unique(y)

# Quick label sanity print
counts = Counter(y)
print("Label counts:", dict(counts))
if len(unique_labels) < 2:
    print("Silhouette score requires at least 2 distinct labels; skipping silhouette.")
    compute_silhouette = False
else:
    compute_silhouette = True

Zs, trusts, sils, sils_by_label_list = [], [], [], []

for s in seeds:
    reducer = umap.UMAP(
        n_neighbors=nearest_neighbours,
        min_dist=min_dist,
        metric="euclidean",
        random_state=s
    )
    Z = reducer.fit_transform(Xn)
    Zs.append(Z)

    # Trustworthiness of the embedding relative to high-D space
    t = trustworthiness(Xn, Z, n_neighbors=trustworthiness_nn, metric="cosine")
    trusts.append(t)

    # Silhouette in the embedding (euclidean distances in 2D), overall & per label
    if compute_silhouette:
        try:
            sil = silhouette_score(Z, y, metric="euclidean")
            s_samples = silhouette_samples(Z, y, metric="euclidean")
            # per-label mean silhouette
            sil_by_label = {lbl: float(np.mean(s_samples[y == lbl])) for lbl in unique_labels}
        except Exception as e:
            sil = np.nan
            sil_by_label = {lbl: np.nan for lbl in unique_labels}
        sils.append(sil)
        sils_by_label_list.append(sil_by_label)

# Stability: how similar are embeddings across seeds? (correlation of pairwise distances vs the first seed)
ref_d = pdist(Zs[0], metric="euclidean")
stabilities = []
for Z in Zs[1:]:
    d = pdist(Z, metric="euclidean")
    r = np.corrcoef(ref_d, d)[0, 1]  # Pearson r over distance vectors
    stabilities.append(r)

print(f"\nTrustworthiness (k={trustworthiness_nn}, metric=cosine) over seeds {seeds}:")
for s, t in zip(seeds, trusts):
    print(f"  seed={s}: {t:.4f}")
print(f"  mean ± std: {np.mean(trusts):.4f} ± {np.std(trusts):.4f}")

if compute_silhouette:
    print(f"\nSilhouette (euclidean in embedding) over seeds {seeds}:")
    for i, s in enumerate(seeds):
        print(f"  seed={s}: overall={sils[i]:.4f}  per-label=" +
              ", ".join([f"{lbl}:{sils_by_label_list[i][lbl]:.3f}" for lbl in unique_labels]))
    print(f"  mean ± std (overall): {np.nanmean(sils):.4f} ± {np.nanstd(sils):.4f}")

print(f"\nEmbedding stability vs seed {seeds[0]} (pairwise-distance correlation):")
for s, r in zip(seeds[1:], stabilities):
    print(f"  seed={s}: r={r:.4f}")
print(f"  mean r: {np.mean(stabilities):.4f}")

# ---------------- Plot labeled embedding (color by call ID) ----------------
Z = Zs[0]

### CHOOSE WHAT TO COLOR BY ###
# (change y to the variable you want, e.g. bird_ids or call_ids)
### BIRDS ###
# plt.figure(figsize=(8, 8))
# interest_variable = np.unique(bird_ids)
# cmap = plt.cm.get_cmap("tab20", len(interest_variable))
# title = "Bird"

### CALL TYPE ###
plt.figure(figsize=(8, 8))
interest_variable = np.unique(y)
cmap = plt.cm.get_cmap("tab20", len(interest_variable))
title = "Call Type"

# ---- Scatter by group ----
for i, variable in enumerate(interest_variable):
    idx = np.where(y == variable)[0]
    plt.scatter(
        Z[idx, 0], Z[idx, 1],
        s=60, color=cmap(i), alpha=0.9, edgecolor="black",
        label=variable
    )

# ---- Add text labels to each point ----
# for i, (coords, label) in enumerate(zip(Z, bird_and_call_ids)):
#     plt.text(
#         coords[0] + 0.6, coords[1], str(label),
#         fontsize=7, ha='center', va='center', alpha=0.7
#     )

# ---- Title and layout ----
title_extra = f" | sil={sils[0]:.3f} (overall)" if compute_silhouette else ""
plt.title(
    f"UMAP (cosine) — seed {seeds[0]} | nn={nearest_neighbours}, min_dist={min_dist}\n"
    f"trust={trusts[0]:.3f} | mean trust={np.mean(trusts):.3f} | "
    f"mean stability r={np.mean(stabilities):.3f}{title_extra}"
)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title=title, markerscale=1.1, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()