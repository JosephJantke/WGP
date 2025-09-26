import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap, hdbscan
matplotlib.use("TkAgg")

# 1) Collect your (already segmented) audio snippets; if not segmented, do that first (see note below)

# AUDIO_DIR = "C:/WGP/UMAP/audio_snippets"  # one file per call/syllable/etc.
AUDIO_DIR = "/UMAP/D_syllables"  #laptop!!
paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

# 2) Convert each snippet to a fixed-shape log-Mel spectrogram vector

SR = 16000
N_FFT = 1024
HOP = 256
N_MELS = 128

# (A) Band-pass target band for WGP contact calls
FMIN, FMAX = 1600, 4000

# Median subtraction strength (1.0 = full median floor removal)  #todo this may have a big impact! Mess around with it
ALPHA = 1.0

# ---------------- Find longest clip -----------------
max_len = 0
waveforms, bird_ids = [], []
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)

    # (B) Amplitude normalization (RMS -> 1.0)
    rms = np.sqrt(np.mean(y**2)) or 1.0
    y = y / rms

    waveforms.append(y)
    max_len = max(max_len, len(y))

    # (Label) take first 10 characters of filename as bird ID
    bird_id = os.path.basename(p)[:10]
    bird_ids.append(bird_id)

    # Print duration
    duration_sec = len(y) / sr
    print(f"{bird_id} | {os.path.basename(p)} : {duration_sec:.2f} seconds")

# Zero-pad all to the same length (no trimming)
waveforms = [np.pad(y, (0, max_len - len(y))) for y in waveforms]
n_frames = int(np.ceil(max_len / HOP))  # expected spectrogram width

# ------------- Feature builder per waveform ----------
def waveform_to_feature(y):
    # (A) Band-pass via Mel limits (in-band spectrogram only)
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )  # power units, shape: (n_mels, T)

    # (Median spectral subtraction) robust per-frequency noise-floor removal
    noise_floor = np.median(S, axis=1, keepdims=True)      # (n_mels, 1)
    S_clean = np.clip(S - ALPHA * noise_floor, 0.0, None)

    # Convert to dB for perceptual scaling
    S_db = librosa.power_to_db(S_clean + 1e-12, ref=np.max)

    # (C) Global z-score standardisation (per clip)
    S_std = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)

    # Defensive pad to exactly n_frames (rounding guard)
    if S_std.shape[1] < n_frames:
        S_std = np.pad(S_std, ((0,0), (0, n_frames - S_std.shape[1])), mode="edge")

    return S_std

# -------------- Build matrix for UMAP ---------------
specs = [waveform_to_feature(y) for y in waveforms]
X = np.vstack([S.ravel() for S in specs])
import math
import librosa.display

def plot_all_spectrograms(specs, file_paths, max_cols=6, cmap="magma"):
    """
    Plot every processed Mel spectrogram in 'specs' as a grid.
    - specs: list/array of 2D arrays (n_mels x T), already processed
    - file_paths: list of file paths aligned with specs
    - max_cols: max number of columns in the grid
    """
    N = len(specs)
    if N == 0:
        print("No spectrograms to plot.")
        return

    cols = min(max_cols, N)
    rows = math.ceil(N / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*2.8), sharex=True, sharey=True)
    # Ensure axes is 2D array for consistent indexing
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    last_img = None
    for i in range(N):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        S = specs[i]

        last_img = librosa.display.specshow(
            S,
            sr=SR,
            hop_length=HOP,
            x_axis="ms",
            y_axis="mel",
            fmin=FMIN,
            fmax=FMAX,
            ax=ax,
            cmap=cmap
        )
        title_prefix = os.path.basename(file_paths[i])[:10]
        ax.set_title(title_prefix, fontsize=9)
        if c == 0:
            ax.set_ylabel("Mel")
        else:
            ax.set_ylabel("")
        if r == rows - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xlabel("")

    # Hide any extra axes
    for j in range(N, rows*cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    # One shared colorbar
    if last_img is not None:
        fig.colorbar(last_img, ax=axes.ravel().tolist(), shrink=0.7, pad=0.01, format="%+2.f dB")

    fig.suptitle("Processed Mel Spectrograms", y=0.995, fontsize=12)
    plt.tight_layout()
    plt.show()

plot_all_spectrograms(specs, paths, max_cols=6)

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
plt.figure(figsize=(7,8))

# Assign unique colors per bird
unique_birds = sorted(set(bird_ids))
cmap = plt.cm.get_cmap("tab20", len(unique_birds))  # brighter, 20-color palette

for i, bird in enumerate(unique_birds):
    idx = [j for j, b in enumerate(bird_ids) if b == bird]
    plt.scatter(Z[idx,0], Z[idx,1],
                s=60,                # slightly larger for visibility
                color=cmap(i),
                alpha=0.9,
                edgecolor="black",   # thin outline helps distinguish points
                label=bird)

plt.title("UMAP of spectrograms by Bird")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.2)
plt.tight_layout()
plt.show()

### this section is if you want to use HDBSCAN

# # 4) Density clustering in the embedding (HDBSCAN)
# clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
# labels = clusterer.fit_predict(Z)  # -1 are "noise"/outliers
#
# # 5) Visualise
# plt.figure(figsize=(7,6))
# palette = np.array(plt.cm.tab20.colors)
# c = np.where(labels < 0, 0, labels % len(palette))  # map to colors
# plt.scatter(Z[:,0], Z[:,1], s=12, c=palette[c], alpha=0.9, linewidths=0)
# plt.title("UMAP of spectrograms + HDBSCAN clusters")
# plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
# plt.show()