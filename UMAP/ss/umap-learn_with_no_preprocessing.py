#no audio preprocessing but has the same time normalisation as in "umap-learn_with_time_normalisation.py"

import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
#no audio preprocessing but has the same time normalisation as in "umap-learn_with_time_normalisation.py"

import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
# AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/D_syllables" #laptop
AUDIO_DIR = "/captive_calls/D_syllables"

paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

SR     = 22000
N_FFT  = 2048
HOP    = 128
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

# ---------------- Build matrix for UMAP ----------------
X = np.vstack([S.ravel() for S in specs_padded])

# ---------------- UMAP ----------------
reducer = umap.UMAP(metric="cosine", random_state=42)
Z = reducer.fit_transform(X)

# ---------------- Plot by bird ----------------
plt.figure(figsize=(8,8))
unique_birds = sorted(set(bird_ids))
cmap = plt.cm.get_cmap("tab20", len(unique_birds))  # bold, distinct colors

for i, bird in enumerate(unique_birds):
    idx = [j for j, b in enumerate(bird_ids) if b == bird]
    plt.scatter(Z[idx,0], Z[idx,1],
                s=60, color=cmap(i), alpha=0.9, edgecolor="black", label=bird)

plt.title("UMAP of log-rescaled (time), padded raw Mel spectrograms by Bird (no preprocessing)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.1)
plt.tight_layout()
plt.show()