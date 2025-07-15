import os
import glob
import soundfile as sf
import math
from pathlib import Path
directory = "E:/SZ673_BlackchinnedHoneyeater/jj_ss"  # todo change this when I move on to the next directory
os.chdir(directory)
home = directory
folder = directory + "/**/*.txt"
def remove_blank_lines(folder):
    for file in glob.glob(folder, recursive=True):
        file = Path(file)
        lines = file.read_text().splitlines()
        # join_path = os.path.join(folder, file)
        # lines = join_path.read_text().splitlines()
        filtered = [
            line
            for line in lines
            if line.strip()
        ]
        file.write_text('\n'.join(filtered))
remove_blank_lines(folder)

for file in glob.glob(folder, recursive=True):
    # print(file)
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    # print(content)

    with open(file, "w") as f:  # todo remember to change this name if necessary
        ### write the first two lines
        f.write(
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate\n")

        for line in content[1:]:
            f.write("%s\n" % line)