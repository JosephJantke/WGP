import torch
import torchaudio
import os
import glob
import shutil
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

#TODO MAKE SURE ALL AUDIO FILES ARE 5 SECONDS EXACTLY

### Function for generating waveforms for melspec conversion

def load_and_preprocess_wav(file_path, target_sr=48000, duration=5.0):
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

    return waveform

### Function for generating melspectrograms

def get_birdnet_v24_spectrograms(waveform):

    power=2.0  #todo see if making melspectrograms with power = 1.0 affects training of model 0

    # LOW BAND PARAMETERS:
    l_sample_rate = 48000
    l_n_fft = 2048
    l_hop_length = 278
    l_n_mels = 96
    l_f_min = 0
    l_f_max = 3000

    # HIGH BAND PARAMETERS:

    h_sample_rate = 48000
    h_n_fft = 1024
    h_hop_length = 280
    h_n_mels = 96
    h_f_min = 500
    h_f_max = 15000

    # Low band: 0–12kHz
    mel_low = MelSpectrogram(
        sample_rate=l_sample_rate,
        n_fft=l_n_fft,
        hop_length=l_hop_length,
        n_mels=l_n_mels,
        f_min=l_f_min,
        f_max=l_f_max,
        power=power
    )(waveform)

    # High band: 12–24kHz
    mel_high = MelSpectrogram(
        sample_rate=h_sample_rate,
        n_fft=h_n_fft,
        hop_length=h_hop_length,
        n_mels=h_n_mels,
        f_min=h_f_min,
        f_max=h_f_max,
        power=power
    )(waveform)

    # Convert to dB scale
    mel_low_db = AmplitudeToDB(top_db=80)(mel_low)    #the value 80 here probably deserves some more research
    mel_high_db = AmplitudeToDB(top_db=80)(mel_high)

    #Ensure output is [3, 96, 511] - 3 because 3 channels necessary for efficientnet backbone, 96 x 511 to match birdnet
    mel_low_db = mel_low_db.squeeze(0).unsqueeze(0).repeat(3, 1, 1)
    mel_high_db = mel_high_db.squeeze(0).unsqueeze(0).repeat(3, 1, 1)

    return mel_low_db, mel_high_db

### Function for ensuring all melspectrograms are 96 x 511 (guessing that they pad/truncate the melspectrograms)

### Create Melspectrums from audio in "wavs_for_mel_conversion"
os.chdir("D:/PhD/WGP_model/from_toshiba_dbca_examples/wavs_for_mel_conversion")
folder = "D:/PhD/WGP_model/from_toshiba_dbca_examples/wavs_for_mel_conversion/**/*.wav"

###segment audio into 5 second snippets if necessary NOTE: DELETES THE ORIGINAL AUDIO FILE
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    # print(file)
    print(file)
    #Processn wave forms
    waveform = load_and_preprocess_wav(file, target_sr=48000, duration=5.0)
    #Generate melspectrograms
    mel1, mel2 = get_birdnet_v24_spectrograms(waveform)
    filename = os.path.splitext(file)[0]

    #save melspectrogram tensors to "wavs_for_mel_conversion" (even though the code says "mel_spectrograms" directory...
    torch.save(mel1, os.path.join("D:/PhD/WGP_model/from_toshiba_dbca_examples/mel_spectrograms/", f"{filename}_mel_low.pt"))
    torch.save(mel2, os.path.join("D:/PhD/WGP_model/from_toshiba_dbca_examples/mel_spectrograms/", f"{filename}_mel_high.pt"))

# all melspectrograms should have a final resolution of 96x511 pixels, CHECK THIS!!!

