import torch
import torchaudio

### convert wav files into melspectrograms
test_wav = "C:/Josephs_PhD/Recogniser/test_set/TEST/1_prob_064_time_00110.3.wav"

waveform, sample_rate = torchaudio.load(test_wav, normalize=True)

#replicate birdnet melspectrogram parameters
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate = 32000,
    n_fft = 2048,
    n_mels = 48,
    hop_length=512,
    f_min = 0,
    f_max = 12000,
    power = 2.0
    )
mel_specgram = transform(waveform)  # (channel, n_mels, time)