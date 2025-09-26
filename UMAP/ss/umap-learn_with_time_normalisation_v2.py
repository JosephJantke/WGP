import os, glob, numpy as np, librosa, librosa.display, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
AUDIO_DIR = "/UMAP/D_syllables"
paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

SR     = 16000
N_FFT  = 1024
HOP    = 256
N_MELS = 128

# Band (Mel limits)
FMIN, FMAX = 1600, 4000

# Median spectral subtraction strength
ALPHA = 1.0

# Duration handling knobs
BETA_PARTIAL = 0.4      # 0 = remove duration; 1 = full duration. Try 0.3–0.5
TRIM_THRESH  = 0.02     # energy threshold for right-trim (fraction of max frame energy)
MIN_FRAMES   = 4        # never trim to fewer than this many frames
Q_TARGET     = 0.95     # cap all lengths to this quantile (crop/pad to this width)

# ---------------- Load waveforms + labels ----------------
waveforms, bird_ids, durs = [], [], []
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)

    # (B) RMS amplitude normalisation
    rms = np.sqrt(np.mean(y**2)) or 1.0
    y = y / rms

    waveforms.append(y)
    durs.append(len(y) / sr)

    bird_id = os.path.basename(p)[:10]
    bird_ids.append(bird_id)

# ---------------- Feature builder (pre-log-rescale) ----------------
def waveform_to_mel_clean(y, return_clean=False):
    # Mel spectrogram in-band (power). center=False avoids extra +/- n_fft//2 padding.
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0, center=False
    )  # (n_mels, T)

    # Median spectral subtraction (per-frequency)
    noise_floor = np.median(S, axis=1, keepdims=True)
    S_clean = np.clip(S - ALPHA * noise_floor, 0.0, None)

    # dB scale + per-clip z-score
    S_db  = librosa.power_to_db(S_clean + 1e-12, ref=np.max)
    S_std = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)

    return (S_std, S_clean) if return_clean else S_std

pairs = [waveform_to_mel_clean(y, return_clean=True) for y in waveforms]
specs_std  = [p[0] for p in pairs]
specs_clean = [p[1] for p in pairs]

# ---------------- Partial log-duration rescaling of time axis ----------------
# Scale by (log-duration / mean-log-duration) ** BETA_PARTIAL
eps = 1e-9
log_durs = np.log(np.array(durs) + eps)
mean_log = float(np.mean(log_durs))
scales_full = log_durs / (mean_log + eps)
scales = np.power(scales_full, BETA_PARTIAL)

def resample_T(S, new_T):
    T = S.shape[1]
    new_T = max(1, int(round(new_T)))
    return librosa.resample(S, orig_sr=T, target_sr=new_T, axis=1)

specs_std_resc, specs_clean_resc = [], []
for S_std, S_cln, sc in zip(specs_std, specs_clean, scales):
    T = S_std.shape[1]
    new_T = max(1, int(round(T * float(sc))))
    specs_std_resc.append(resample_T(S_std, new_T))
    specs_clean_resc.append(resample_T(S_cln, new_T))

# ---------------- Right-trim trailing empty columns ----------------
def right_trim_idx(S_clean, rel_thresh=TRIM_THRESH, min_frames=MIN_FRAMES):
    e = S_clean.mean(axis=0)           # frame-wise energy
    m = e.max() + 1e-12
    active = e > (rel_thresh * m)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return min_frames
    return int(idx[-1] + 1)            # end index (exclusive)

trimmed_std, trimmed_lengths = [], []
for S_std, S_cln in zip(specs_std_resc, specs_clean_resc):
    end = right_trim_idx(S_cln)
    trimmed_std.append(S_std[:, :end])
    trimmed_lengths.append(end)

# ---------------- Cap to a quantile width (light pad/crop) ----------------
T_target = int(np.quantile(trimmed_lengths, Q_TARGET))
T_target = max(T_target, MIN_FRAMES)   # safety

specs_fixed = []
for S in trimmed_std:
    if S.shape[1] >= T_target:
        specs_fixed.append(S[:, :T_target])                         # crop long ones
    else:
        pad = T_target - S.shape[1]
        specs_fixed.append(np.pad(S, ((0,0),(0,pad)), 'constant'))  # light pad short ones

# ---------------- Build matrix for UMAP ----------------
# (No explicit duration feature appended — per your comment)
X = np.vstack([S.ravel() for S in specs_fixed])

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

plt.title("UMAP of Mel spectrograms (partial time-normalisation, right-trimmed, p95 width)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.1, bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()

# (Optional) quick glance at a few panels
# plot_example_spectrograms(specs_fixed, bird_ids, n=6)