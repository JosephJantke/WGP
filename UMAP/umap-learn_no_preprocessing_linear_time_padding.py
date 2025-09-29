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

# ---------------- UMAP on padded Mels ----------------
X = np.vstack([S.ravel() for S in specs_padded])   # (N, N_MELS*T_max)
reducer = umap.UMAP(metric="cosine", random_state=42)
Z = reducer.fit_transform(X)

# ---------------- UMAP scatter by bird ----------------
plt.figure(figsize=(8, 8))
unique_birds = sorted(set(bird_ids))
cmap = plt.cm.get_cmap("tab20", len(unique_birds))
for i, bird in enumerate(unique_birds):
    idx = [j for j, b in enumerate(bird_ids) if b == bird]
    plt.scatter(Z[idx, 0], Z[idx, 1], s=60, color=cmap(i), alpha=0.9,
                edgecolor="black", label=bird)

plt.title("UMAP of padded raw Mel spectrograms (no log time rescale)")
plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
plt.margins(x=0.2, y=0.2)
plt.legend(title="Bird ID", markerscale=1.1, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout(); plt.show()