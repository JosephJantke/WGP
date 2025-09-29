

# no audio preprocessing; keep log-time normalisation (resample Mel along time), then pad to longest
import os, glob, numpy as np, librosa, matplotlib, matplotlib.pyplot as plt, umap
matplotlib.use("TkAgg")

# ---------------- Config ----------------
# AUDIO_DIR = "C:/Users/a1801526/PycharmProjects/WGP_laptop/UMAP/D_syllables"  # laptop
AUDIO_DIR = "/captive_calls/rising_step_syllables"

paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))

SR     = 22000
N_FFT  = 2048
HOP    = 128
N_MELS = 128
FMIN, FMAX = 2000, 3500
POWER  = 0.5  # root compression

# ---------------- Load waveforms + labels (no amplitude/RMS normalisation) ----------------
waveforms, bird_ids, durs = [], [], []

print("File durations:")
for p in paths:
    y, sr = librosa.load(p, sr=SR, mono=True)
    dur_s = len(y) / sr
    print(f"  {os.path.basename(p):<40} {dur_s*1000:7.1f} ms  ({dur_s:.3f} s)")

    waveforms.append(y)
    durs.append(dur_s)
    bird_ids.append(os.path.basename(p)[:10])

# summary
if durs:
    d = np.array(durs)
    print(f"\nTotal files: {len(durs)} | mean: {d.mean()*1000:.1f} ms | median: {np.median(d)*1000:.1f} ms | "
          f"min: {d.min()*1000:.1f} ms | max: {d.max()*1000:.1f} ms\n")

# ---------------- Raw Mel (no dB/z-score; avoid STFT auto-padding) ----------------
def waveform_to_mel_raw(y):
    return librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        power=POWER, center=False
    )

specs_raw = [waveform_to_mel_raw(y) for y in waveforms]

# ---------------- Log-duration rescaling of time axis ----------------
eps = 1e-9
log_durs = np.log(np.array(durs) + eps)
mean_log = log_durs.mean()
scales = np.exp(log_durs - mean_log)   # == durations / geometric_mean(durations)

specs_rescaled, rescaled_lengths = [], []
for S, scale in zip(specs_raw, scales):
    T = S.shape[1]
    new_T = max(1, int(np.round(T * float(scale))))
    S_resc = librosa.resample(S, orig_sr=T, target_sr=new_T, axis=1)
    specs_rescaled.append(S_resc)
    rescaled_lengths.append(new_T)

# ---------------- Zero-pad to longest log-rescaled length ----------------
max_rescaled_T = int(max(rescaled_lengths)) if rescaled_lengths else 0
specs_padded = []
for S in specs_rescaled:
    if S.shape[1] < max_rescaled_T:
        S = np.pad(S, ((0,0), (0, max_rescaled_T - S.shape[1])), mode="constant")
    else:
        S = S[:, :max_rescaled_T]
    specs_padded.append(S)

# ---------------- UMAP ----------------
X = np.vstack([S.ravel() for S in specs_padded]) if specs_padded else np.empty((0,0))
reducer = umap.UMAP(metric="cosine", random_state=42)
Z = reducer.fit_transform(X) if X.size else np.empty((0,2))

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
plt.legend(title="Bird ID", markerscale=1.1, bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.show()