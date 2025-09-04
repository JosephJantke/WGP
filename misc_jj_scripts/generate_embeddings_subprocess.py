# generate_embeddings_subprocess.py
import os, sys, subprocess
import torch, torchaudio, os, glob, shutil
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

### for background files

# audio_input       = r"D:/PhD\WGP_model/background_recordings/recordings"
# embeddings_output = r"D:/PhD\WGP_model/background_recordings/fp_embeddings"
# db_path           = os.path.join(embeddings_output, "embeddings.sqlite")   #don't need the sqlite file, but need to satisfy function

### for wgp recordings

audio_input       = r"D:/PhD/WGP_model/toshiba_dbca_examples_tps/CANP_monitoring_ARU_recordings"
embeddings_output = r"D:/PhD/WGP_model/toshiba_dbca_examples_tps/tp_embeddings"
db_path           = os.path.join(embeddings_output, "embeddings.sqlite")   #don't need the sqlite file, but need to satisfy function


### preprocess wav files:

#function:
def load_and_preprocess_wav(file_path, target_sr=48000):
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 48kHz
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    # Normalize to [-1, 1]
    waveform = waveform / waveform.abs().max()

    torchaudio.save(file_path, waveform, target_sr, encoding="PCM_S", bits_per_sample=16)

    return waveform

folder = audio_input + "/**/*.wav"
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    # print(file)
    print(file)
    #Process wave forms
    waveform = load_and_preprocess_wav(file, target_sr=48000)

os.makedirs(embeddings_output, exist_ok=True)

cmd = [
    sys.executable, "-m", "birdnet_analyzer.embeddings",
    "-i", audio_input,             # <-- use -i for input path
    "-t", "8",                     # <-- short flag per usage ([-t THREADS])
    "-b", "16",
    "-db", db_path,
    "--fmin", "1000",
    "--fmax", "8000",
    "--overlap", "0.0",
    "--file_output", embeddings_output,
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

#this will generate embeddings as numpy arras — may need to convert into csvs
