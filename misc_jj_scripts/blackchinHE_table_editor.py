from tqdm import tqdm
import re
import os
import PIL
import shutil
import glob
import csv
import soundfile as sf

###########delete two blank lines at the end of each text file
from pathlib import Path
import os
import glob
home = "E:/SZ673_BlackchinnedHoneyeater/jj_ss"
folder = home + "/**/*.txt"
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

#################change first line of each text file
import os
import glob
directory = "E:/SZ673_BlackchinnedHoneyeater/jj_ss"  #todo change this when i want to work on different directories
os.chdir(directory)
home = directory
folder = directory + "/**/*.txt"
for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    #print(content[0])
    # print(content)
    if content[0] != "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate":
        # print("yipee")
        content[0] = "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate"
    with open(file, "w") as f:
        for line in content:
            # print(line)
            f.write("%s\n" % line)

##############Change the first line and the high frequency to 16000 Hz
import os
import glob

directory = "E:/SZ673_BlackchinnedHoneyeater/jj_ss"  #todo change this when i want to work on different directories
os.chdir(directory)
home = directory
folder = directory + "/**/*.txt"
for file in glob.glob(folder, recursive=True):
    print(file)
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    # content_fragmented = content.split("\t")
    # print(content)

    with open(file + "_JJ.txt", "w") as f:
        f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate\n")
        for line in content[1:]:
            line_split = line.split("\t")
            # print(line_split[6])
            if line_split[6] != "16000":
                line_split[6] = "16000"
            line_rejoined = '\t'.join(line_split)
            # print(line_rejoined)
            f.write("%s\n" % line_rejoined)

# change the duration to 5.0 and reduce end time by 0.1 seconds
import os
import glob
directory = "E:/SZ673_BlackchinnedHoneyeater/jj_ss"  #todo change this when i want to work on different directories
os.chdir(directory)
home = directory
folder = directory + "/**/*.txt"
for file in glob.glob(folder, recursive=True):
    print(file)
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    # content_fragmented = content.split("\t")
    # print(content)
    with open(file + "_JJ.txt", "w") as f:
        f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate\n")
        for line in content[1:]:
            line_split = line.split("\t")
            # print(line_split[8])
            line_split[4] = float(line_split[4])
            line_split[4] = line_split[4] - 0.1
            line_split[4] = round(line_split[4], 3)
            line_split[4] = str(line_split[4])
            if line_split[8] != "5.0":
                line_split[8] = "5.0"
            line_rejoined = '\t'.join(line_split)
            # print(line_rejoined)
            f.write("%s\n" % line_rejoined)



#todo this code enters a 1 or 0 into the validation column of the text file
os.chdir("E:/joseph/model_x/inference/model_2/selection_tables")
home = "E:/joseph/model_x/inference/model_2/selection_tables"
folder = "E:/joseph/model_x/inference/model_2/selection_tables/**/*.txt"
for file in glob.glob(folder, recursive=True):
    # print(file)
    textfilename = file[55:78]
    csvfilename = file[53:78] + '.csv'
    print(textfilename)
    # print(csvfilename)
    join_path = os.path.join(folder, file)
    f = open(join_path, 'r')
    content = f.read()
    content = content.split("\n")
    # print(len(content))

