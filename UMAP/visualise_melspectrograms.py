import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

#todo the code between here and "visualise spectrograms if necessary" needs to be taken from the "umap-learn..." scripts
# Simple pipeline: no preprocessing, no log time rescale.
# Pad Mel spectrograms to the longest one; keep plot time scale normal (ms).
import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
# AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/D_syllables"  # laptop
AUDIO_DIR = "C:/WGP/captive_calls/rising_step_syllables"

paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
if not paths:
    raise SystemExit(f"No .wav files found in: {AUDIO_DIR}")

SR     = 22000
N_FFT  = 2048
HOP    = 128
N_MELS = 128
FMIN, FMAX = 2000, 3500

# Root compression for a touch more sensitivity (features only; still "no preprocessing")
POWER  = 0.7   # try 0.5–0.7; 1.0 = amplitude, 2.0 = power

# ---------------- Load waveforms + labels (no RMS/dB/z-score) ----------------
waveforms, bird_ids, durs = [], [], []

print("File durations:")
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)
    dur_s = len(y) / sr
    print(f"  {os.path.basename(p):<40} {dur_s*1000:7.1f} ms  ({dur_s:.3f} s)")
    waveforms.append(y)
    bird_ids.append(os.path.basename(p)[:10])
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





# -------- Visualise melspectrograms if necessary --------

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


plot_all_spectrograms(specs_padded, paths, max_cols=6)