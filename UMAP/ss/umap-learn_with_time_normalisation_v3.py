# duration not weighted in clustering, but call should occupy ~90% of each Mel panel
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

# Fill/occupancy target
OCC_TARGET      = 0.90   # call should fill ~90% of each panel
CONTENT_THRESH  = 0.02   # fraction of max frame energy to consider "active"
MIN_FRAMES_CALL = 4      # safety minimum

# Choose a robust common "content length" after your log-time normalisation.
# Median is conservative (less extreme stretching); 0.75–0.9 quantile is also fine.
CONTENT_TARGET_STAT = "median"   # or "q90"
Q = 0.90

# ---------------- Load waveforms + labels ----------------
waveforms, bird_ids, durs = [], [], []
for p in paths:
    y, _sr = librosa.load(p, sr=SR, mono=True)
    # RMS amplitude normalisation
    rms = np.sqrt(np.mean(y**2)) or 1.0
    y = y / rms

    waveforms.append(y)
    durs.append(len(y) / SR)
    bird_ids.append(os.path.basename(p)[:10])

durs = np.asarray(durs)

# ---------------- Helper: Mel (cleaned) + "active" length ----------------
def mel_and_clean(y):
    # center=True is the default; it can add ~n_fft/2 pad at both ends, but keeps crisp visuals.
    S = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0
    )  # (n_mels, T)

    # median spectral subtraction (per frequency)
    noise_floor = np.median(S, axis=1, keepdims=True)
    S_clean = np.clip(S - ALPHA * noise_floor, 0.0, None)

    # dB + z-score (for plotting/ML)
    S_db  = librosa.power_to_db(S_clean + 1e-12, ref=np.max)
    S_std = (S_db - np.mean(S_db)) / (np.std(S_db) + 1e-8)
    return S_std, S_clean

def last_active_frame(S_clean, rel=CONTENT_THRESH):
    # frame energy as mean across mel bands
    e = S_clean.mean(axis=0)  # (T,)
    m = float(e.max()) + 1e-12
    active = e > (rel * m)
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return MIN_FRAMES_CALL
    return int(idx[-1] + 1)  # exclusive end

# ---------------- First pass: measure content lengths & base scales ----------------
# Your original log-duration scale
eps = 1e-9
log_durs = np.log(durs + eps)
mean_log = float(np.mean(log_durs))
base_scales = log_durs / (mean_log + eps)   # same idea you had

content_frames_orig = []
for y in waveforms:
    _, S_clean0 = mel_and_clean(y)
    content_frames_orig.append(last_active_frame(S_clean0))
content_frames_orig = np.asarray(content_frames_orig, dtype=int)

# What would the content be *after* your log-duration scale?
content_frames_scaled_base = content_frames_orig * base_scales

# Pick a common target content length (in frames) after scaling
if CONTENT_TARGET_STAT.lower() == "q90":
    F_content = int(np.ceil(np.quantile(content_frames_scaled_base, Q)))
else:
    F_content = int(np.ceil(np.median(content_frames_scaled_base)))
F_content = max(F_content, MIN_FRAMES_CALL)

# Given we want each panel to have OCC_TARGET occupancy, set the final common width:
T_final = int(np.ceil(F_content / OCC_TARGET))

# ---------------- Second pass: single-pass waveform time-scaling, then Mel ----------------
specs_final = []
for y, s1, c0 in zip(waveforms, base_scales, content_frames_orig):
    # Combined scale so that content ≈ F_content frames after scaling:
    # content_after ≈ c0 * s1 * gamma = F_content  =>  gamma = F_content / (c0 * s1)
    gamma = float(F_content) / (float(c0) * float(s1) + 1e-12)
    s_total = s1 * gamma

    # librosa.effects.time_stretch uses 'rate' where new_dur = old_dur / rate
    rate = 1.0 / max(s_total, 1e-6)

    # Guard very short clips: pad a tad before stretching if needed
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)), mode="constant")

    y_stretch = librosa.effects.time_stretch(y, rate=rate)

    # Now compute Mel at full resolution (no extra resamples of the Mel!)
    S_std, S_clean = mel_and_clean(y_stretch)

    # Recompute actual content after stretch, then minimally crop/pad to T_final
    c_after = last_active_frame(S_clean)
    # Crop only if longer than T_final (but keep content!)
    if S_std.shape[1] > T_final:
        S_std = S_std[:, :T_final]
    # Pad if shorter
    if S_std.shape[1] < T_final:
        pad = T_final - S_std.shape[1]
        S_std = np.pad(S_std, ((0,0),(0,pad)), mode="constant")

    specs_final.append(S_std)

specs_final = np.asarray(specs_final)  # (N, n_mels, T_final)

# ---------------- Build matrix for UMAP ----------------
X = np.vstack([S.ravel() for S in specs_final])

# ---------------- UMAP ----------------
reducer = umap.UMAP(metric="cosine", random_state=42)
Z = reducer.fit_transform(X)

# ---------------- Plot by bird ----------------
plt.figure(figsize=(8,8))
unique_birds = sorted(set(bird_ids))
cmap = plt.cm.get_cmap("tab20", len(unique_birds))

for i, bird in enumerate(unique_birds):
    idx = [j for j, b in enumerate(bird_ids) if b == bird]
    plt.scatter(Z[idx,0], Z[idx,1], s=60, color=cmap(i),
                alpha=0.9, edgecolor="black", label=bird)

plt.title(f"UMAP of Mel spectrograms (log-time norm + 90% fill, T={T_final})")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.1, bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()

# (Optional) sanity check: occupancy stats
occupancies = []
for S in specs_final:
    # estimate active frames again on z-scored Mel (okay for a quick check)
    e = np.maximum(S, 0).mean(axis=0)
    m = e.max() + 1e-12
    c = int(np.flatnonzero(e > CONTENT_THRESH * m)[-1] + 1) if np.any(e > CONTENT_THRESH*m) else MIN_FRAMES_CALL
    occupancies.append(c / T_final)
print(f"Occupancy: min={np.min(occupancies):.3f}, median={np.median(occupancies):.3f}, "
      f"mean={np.mean(occupancies):.3f}, target≈{OCC_TARGET:.2f}")