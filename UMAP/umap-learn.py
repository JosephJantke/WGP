import os, glob, numpy as np, librosa, matplotlib.pyplot as plt
import umap
import hdbscan

# 1) Collect your (already segmented) audio snippets; if not segmented, do that first (see note below)
AUDIO_DIR = "C:/WGP/UMAP/audio_snippets"  # one file per call/syllable/etc.
paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

# 2) Convert each snippet to a fixed-shape log-Mel spectrogram vector
SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 128
FMIN = 150
DUR_S = 3.0  # force a consistent window length (pad/trim)
n_frames = int(np.ceil(DUR_S * SR / HOP))

def wav_to_vec(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    # pad/trim to DUR_S seconds so all spectrograms are the same shape
    target_len = int(DUR_S * SR)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # log-Mel spectrogram (you can also use linear STFT magnitude)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=FMIN, fmax=sr/2)
    S_db = librosa.power_to_db(S, ref=np.max)

    # force exact frame count (defensive)
    if S_db.shape[1] < n_frames:
        S_db = np.pad(S_db, ((0,0),(0, n_frames - S_db.shape[1])), mode="edge")
    else:
        S_db = S_db[:, :n_frames]

    # per-sample standardisation helps; cosine metric is also robust
    S_db = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    return S_db.ravel()

X = np.vstack([wav_to_vec(p) for p in paths])

# 3) UMAP embedding
reducer = umap.UMAP(
    # n_neighbors=30,      # try 15–50
    # min_dist=0.1,        # smaller -> tighter clusters
    # n_components=2,
    # metric="cosine",     # cosine is good for spectrograms!!
    # random_state=42
)
Z = reducer.fit_transform(X)   # shape (n_samples, 2)

# 4) Plot UMAP embedding with a scatterplot
plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], s=10, alpha=0.8, c="steelblue", linewidths=0)
plt.title("UMAP of spectrograms (default params)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()


### this section is if you want to use HDBSCAN

# 4) Density clustering in the embedding (HDBSCAN)
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
labels = clusterer.fit_predict(Z)  # -1 are "noise"/outliers

# 5) Visualise
plt.figure(figsize=(7,6))
palette = np.array(plt.cm.tab20.colors)
c = np.where(labels < 0, 0, labels % len(palette))  # map to colors
plt.scatter(Z[:,0], Z[:,1], s=12, c=palette[c], alpha=0.9, linewidths=0)
plt.title("UMAP of spectrograms + HDBSCAN clusters")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.show()