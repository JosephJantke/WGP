import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# -------- Config (higher resolution) --------
AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/rising_step_syllables"
paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

SR     = 16000
N_FFT  = 512          # was 1024 -> shorter window (~32 ms)
HOP    = 64           # was 256  -> finer time (~4 ms)
N_MELS = 256          # was 128  -> more freq detail
FMIN, FMAX = 1600, 4000

# -------- Load (no amplitude normalisation) --------
waveforms, bird_ids, durs = [], [], []
for p in paths:
    y, _ = librosa.load(p, sr=SR, mono=True)
    waveforms.append(y)
    durs.append(len(y) / SR)
    bird_ids.append(os.path.basename(p)[:10])
durs = np.asarray(durs)

# -------- Time-normalisation factor (your original idea) --------
eps = 1e-9
log_durs = np.log(durs + eps)
mean_log = float(np.mean(log_durs))
scales = log_durs / (mean_log + eps)   # >1 stretch, <1 compress

# -------- Build Mel AFTER waveform time-stretch (preserves detail) --------
def mel_power(y):
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )

specs, lengths = [], []
for y, s in zip(waveforms, scales):
    # librosa time_stretch uses rate where new_len = old_len / rate
    rate = 1.0 / max(float(s), 1e-6)
    if len(y) < N_FFT:  # tiny safety pad for very short clips
        y = np.pad(y, (0, N_FFT - len(y)))
    y_stretch = librosa.effects.time_stretch(y, rate=rate)
    S = mel_power(y_stretch)
    specs.append(S)
    lengths.append(S.shape[1])

# -------- Pad/crop to a common width (light touch) --------
T_target = max(lengths)  # or int(np.quantile(lengths, 0.95)) to cut extreme tails
specs_padded = []
for S in specs:
    T = S.shape[1]
    if T < T_target:
        S = np.pad(S, ((0,0),(0, T_target - T)), mode="constant")
    elif T > T_target:
        S = S[:, :T_target]
    specs_padded.append(S)

# -------- UMAP input --------
X = np.vstack([S.ravel() for S in specs_padded])
reducer = umap.UMAP(metric="cosine", random_state=42)
Z = reducer.fit_transform(X)

# -------- Plot by bird (dB only for viewing — not used for UMAP) --------
fig, axes = plt.subplots(3, 6, figsize=(16,7), sharex=True, sharey=True)
axes = axes.ravel()
for i, ax in enumerate(axes[:min(len(specs_padded), len(axes))]):
    SdB = librosa.power_to_db(specs_padded[i] + 1e-12, ref=np.max)
    librosa.display.specshow(SdB, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel",
                             fmin=FMIN, fmax=FMAX, ax=ax, cmap="magma")
    ax.set_title(bird_ids[i][:10], fontsize=9)
    ax.set_xlabel("Time (s)")
    if i % 6 == 0: ax.set_ylabel("Mel")
plt.suptitle("High-res Mel (waveform time-stretch; no RMS/z-score)", y=0.98)
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,8))
uniq = sorted(set(bird_ids)); cmap = plt.cm.get_cmap("tab20", len(uniq))
for i, b in enumerate(uniq):
    idx = [j for j,x in enumerate(bird_ids) if x == b]
    plt.scatter(Z[idx,0], Z[idx,1], s=60, color=cmap(i), edgecolor="black", alpha=0.9, label=b)
plt.title("UMAP — high-res Mel (no preprocessing)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2"); plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.show()