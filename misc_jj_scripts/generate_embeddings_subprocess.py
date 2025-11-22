# generate_embeddings_subprocess.py
import os, sys, subprocess
import torch, torchaudio, os, glob, shutil
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn.functional as F

### for wgp recordings

audio_input       = r"C:/WGP/captive_calls/rising_step_syllables_without_jinnung_no_whitespaces"
embeddings_output = r"C:/WGP/captive_calls/embeddings/rising_step_syllables_without_jinnung"
db_path           = os.path.join(embeddings_output, "embeddings.sqlite")   #don't need the sqlite file, but need to satisfy function

### replace whitespaces with underscore if necessary
object_for_removing_whitespaces    = audio_input + "/**/*.wav"
for file in glob.glob(object_for_removing_whitespaces, recursive=True):
    join_path = os.path.join(object_for_removing_whitespaces, file)
    print(file)
    destination = file.replace(' ', '_')
    if file != destination:
        shutil.copy(file, destination)
        os.remove(file)

### preprocess wav files:

#function:
def load_and_preprocess_wav(file_path, target_sr=48000, target_len_s=3.0):
    wav, sr = torchaudio.load(file_path)

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)

    # Normalise BEFORE padding
    if wav.abs().max() > 0:
        wav = wav / wav.abs().max()

    # Pad or trim to fixed length
    n_target = int(target_sr * target_len_s)
    if wav.shape[1] < n_target:
        pad = n_target - wav.shape[1]
        left = pad // 2
        right = pad - left
        wav = F.pad(wav, (left, right))  # center the call
    elif wav.shape[1] > n_target:
        wav = wav[:, :n_target]

    torchaudio.save(file_path, wav, target_sr, encoding="PCM_S", bits_per_sample=16)
    return wav

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
    "--fmin", "2000",                                                                                                       #todo remember to change this if necessary
    "--fmax", "4000",                                                                                                       #todo remember to change this if necessary
    "--overlap", "0.0",
    "--file_output", embeddings_output,
]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

#this will generate embeddings as numpy arras — may need to convert into csvs
