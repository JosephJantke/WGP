import wave
import glob
import os

#code for merging wav files
os.chdir('C:/Users/a1801526/PycharmProjects/WGP_laptop/TEST')
folder = r"C:/Users/a1801526/PycharmProjects/WGP_laptop/TEST/**/*.wav*"   ### apparently the r in front of string is important!!

outfile = "test_long.wav"

#glob method (doesn't work on uni laptop)
wav_list = []
data = []
for file in glob.glob(folder, recursive=True):
    # wav_list.append(file)
    print(file)

for file in wav_list:
    w = wave.open(file, 'rb')
    data.append([w.getparams(), w.readframes(w.getnframes())])
    w.close()

output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
for i in range(len(data)):
    output.writeframes(data[i][1])
output.close()


#wav code
infiles = ["sound_1.wav", "sound_2.wav"]
outfile = "sounds.wav"

data = []
for infile in infiles:
    w = wave.open(infile, 'rb')
    data.append([w.getparams(), w.readframes(w.getnframes())])
    w.close()

output = wave.open(outfile, 'wb')
output.setparams(data[0][0])
for i in range(len(data)):
    output.writeframes(data[i][1])
output.close()