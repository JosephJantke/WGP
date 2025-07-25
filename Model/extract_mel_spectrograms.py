import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

#TODO MAKE SURE ALL AUDIO FILES ARE 5 SECONDS EXACTLY

### Generating waveforms for melspec converstion

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

### Generating melspectrograms

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

    return mel_low_db, mel_high_db

# todo both spectrograms should have a final resolution of 96x511 pixels

