import torch
import torchaudio

test_wav = "C:/Josephs_PhD/Recogniser/test_set/TEST/1_prob_064_time_00110.3.wav"

waveform, sample_rate = torchaudio.load(test_wav, normalize=True)
transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels = 64)
mel_specgram = transform(waveform)  # (channel, n_mels, time)