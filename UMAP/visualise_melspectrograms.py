import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

#todo the code between here and "visualise spectrograms if necessary" needs to be taken from the "umap-learn..." scripts
# ---------------- Config ----------------
#no audio preprocessing but has the same time normalisation as in "umap-learn_with_time_normalisation.py"

import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
#no audio preprocessing but has the same time normalisation as in "umap-learn_with_time_normalisation.py"

import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/D_syllables"
paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

SR     = 22000
N_FFT  = 1024
HOP    = 256
N_MELS = 128

# Band (Mel limits)
FMIN, FMAX = 2000, 3500

# ---------------- Load waveforms + labels (no amplitude/RMS normalisation) ----------------
max_len = 0
waveforms, bird_ids, durs = [], [], []
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)

    waveforms.append(y)
    durs.append(len(y) / sr)
    max_len = max(max_len, len(y))

    bird_id = os.path.basename(p)[:10]
    bird_ids.append(bird_id)

# Zero-pad all waveforms to same length (no trimming)
waveforms = [np.pad(y, (0, max_len - len(y))) for y in waveforms]

# ---------------- Feature builder (NO subtraction / NO dB / NO z-score) ----------------
def waveform_to_mel_raw(y):
    # Returns raw Mel power spectrogram (float) with no additional processing
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=0.5
    )  # shape: (n_mels, T)
    return S

# Compute Mel specs (pre time-rescale)
specs_raw = [waveform_to_mel_raw(y) for y in waveforms]

# ---------------- Log-duration rescaling of time axis (as originally implemented) ----------------
eps = 1e-9
log_durs = np.log(np.array(durs) + eps)
mean_log = float(np.mean(log_durs))

# Scale factor for each clip: bigger than 1 → stretch; smaller → compress
scales = log_durs / (mean_log + eps)

# Resample each spectrogram along time (axis=1) by its scale
specs_rescaled = []
rescaled_lengths = []
for S, scale in zip(specs_raw, scales):
    T = S.shape[1]
    new_T = max(1, int(np.round(T * float(scale))))
    S_resc = librosa.resample(S, orig_sr=T, target_sr=new_T, axis=1)
    specs_rescaled.append(S_resc)
    rescaled_lengths.append(new_T)

# ---------------- Zero-pad to longest log-rescaled length ----------------
max_rescaled_T = int(max(rescaled_lengths))
specs_padded = []
for S in specs_rescaled:
    if S.shape[1] < max_rescaled_T:
        S_pad = np.pad(S, ((0,0),(0, max_rescaled_T - S.shape[1])), mode="constant")
    else:
        S_pad = S[:, :max_rescaled_T]
    specs_padded.append(S_pad)




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