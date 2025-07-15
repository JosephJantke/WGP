### THIS STRETCH OF CODE DOES the following for the data given by Kyle 11/01/2025:
# removes whitespaces
# changes the first line, changes the frequencies from 16000 to 22000, and changes start and end times
import os
import glob
import soundfile as sf
import math
from pathlib import Path
directory = "E:/SZ673_BlackchinnedHoneyeater/jj11-20"  # todo change this when I move on to the next directory
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

    #### make a list with only odd entries from the text file (eg. lines 1, 3, 5, 7 etc.)
    content_odd = []
    for i in range(len(content[1:])):
        if i % 2 == 1:
            content_odd.append(content[i])

    ##set the first line of entries (second line of file) to 0 and 5.056 and duration 5.056
    second_line_fragmented = content[1].split("\t")
    second_line_fragmented[3] = str(0)  #str() is necessary to turn floats back into strings for .join()
    second_line_fragmented[4] = str(5.056)
    second_line_fragmented[8] = str(5.056)
    second_line_fragmented[6] = '16000'
    second_line_rejoined = '\t'.join(second_line_fragmented)
    ##set the second line of entries (third line of file) to 5.152 and 10.216 and duration 5.064
    third_line_fragmented = content[2].split("\t")
    third_line_fragmented[3] = str(5.152)  #str() is necessary to turn floats back into strings for .join()
    third_line_fragmented[4] = str(10.216)
    third_line_fragmented[6] = '16000'
    third_line_fragmented[8] = str(5.064)
    third_line_rejoined = '\t'.join(third_line_fragmented)

    ### write the text file
    with open(file[:-4] + "_real.txt", "w") as f:    #todo remember to change this name if necessary

    ### write the first two lines
        f.write("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tFilename\tdur\tValidate")
        f.write("\n" + second_line_rejoined)# + "\n")
        f.write("\n" + third_line_rejoined + "\n")

    ### write the rest of the entries - alternating in start, end, and duration
        for line in content_odd[1:]:
            lf = line.split("\t")

            ### odd entry:
            lf[0] = float(lf[0]) # convert all strings to floats
            lf[3] = float(lf[3])
            lf[4] = float(lf[4])
            lf[6] = '16000'
            line_no = int(lf[0])  # create this variable for writing even entries in below
            lf[3] = (lf[0]-1)/2 * 10.312  # enter correct values
            lf[4] = lf[3] + 5.056
            lf[3] = round(lf[3], 3)  # use round() to prevent long strings of decimal places
            lf[4] = round(lf[4], 3)
            lf[3] = str(lf[3])  # convert floats to string to be able to rejoin the line into a single string
            lf[4] = str(lf[4])
            lf[0] = int(lf[0])  # convert first entry to int
            lf[0] = str(lf[0])  # convert first entry back to string
            lf[8] = str(5.056)  # convert duration to correct value
            line_rejoined = '\t'.join(lf)  #rejoin string
            f.write("%s\n" % line_rejoined) #todo un-hash this entry and all other write entries when finished

            ### even entry:
            lf = content[line_no+1].split("\t")
            lf[0] = float(lf[0]) # convert all strings to floats
            lf[3] = float(lf[3])
            lf[4] = float(lf[4])
            lf[6] = '16000'
            line_no = int(lf[0])  # create this variable for writing even entries in below
            lf[3] = 15.464 + (((lf[0])/2 - 2) * 10.312)  # enter correct values
            lf[4] = lf[3] + 5.064
            lf[3] = round(lf[3], 3)  # use round() to prevent long strings of decimal places
            lf[4] = round(lf[4], 3)
            lf[3] = str(lf[3])  # convert floats to string to be able to rejoin the line into a single string
            lf[4] = str(lf[4])
            lf[0] = int(lf[0])  # convert first entry to int
            lf[0] = str(lf[0])  # convert first entry back to string
            lf[8] = str(5.064)  # convert duration to correct value
            line_rejoined = '\t'.join(lf)  # rejoin string
            f.write("%s\n" % line_rejoined)


### this is a much older piece of code for changing the start and end duration of audio chunks (OBSOLETE):

# import os
# import glob
# import soundfile as sf
# import math
# os.chdir("E:/SZ673_BlackchinnedHoneyeater/jj/jj_snippets_for_editing_selection_tables")  # todo change this when i want to manipulate the real folder
# home = "E:/SZ673_BlackchinnedHoneyeater/jj/jj_snippets_for_editing_selection_tables"
# folder = "E:/SZ673_BlackchinnedHoneyeater/jj/jj_snippets_for_editing_selection_tables/**/*.txt"
# for file in glob.glob(folder, recursive=True):
#     # print(file)
#     join_path = os.path.join(folder, file)
#     f = open(join_path, 'r')
#     content = f.read()
#     content = content.split("\n")
#     print(content)
#
#     ### Audio Stuff ###
#     #FOR GETTING THE LENGTH OF THE AUDIO_LONG
#     audio_file = file[:71] + "_long.wav"       #todo remember to change this when working on real data
#     audio = sf.SoundFile(audio_file)
#     audio_length = len(audio) / audio.samplerate
#     # print(file + ":   " + str(audio_length))
#
#     ### Changing second line of text file to always have 0 and 5.0 second start and end (and duration!) ###
#     second_line_fragmented = content[1].split("\t")
#     second_line_fragmented[3] = str(0)  #str() is necessary to turn floats back into strings for .join()
#     second_line_fragmented[4] = str(5.0)
#     second_line_fragmented[8] = str(5.0)
#     second_line_rejoined = '\t'.join(second_line_fragmented)
#     with open(file + "_trial.txt", "w") as f:    #todo remember to change this to just file to file
#         f.write(content[0])
#         f.write("\n" + second_line_rejoined + "\n")
#         for line in content[2:]:
#             line_fragmented = line.split("\t")
#             # print(line_fragmented)
#             # start_time = line_fragmented[3]
#             # end_time = line_fragmented[4]
#             line_fragmented[3] = (float(line_fragmented[0])-1)*5.1
#             line_fragmented[4] = 5 + (5.1*(float(line_fragmented[0])-1))
#             line_fragmented[3] = round(line_fragmented[3], 2)
#             line_fragmented[4] = round(line_fragmented[4], 2)
#             line_fragmented[3] = str(line_fragmented[3])  #this is necessary as floats sometimes result in numbers with large decimal places
#             line_fragmented[4] = str(line_fragmented[4])
#             line_fragmented[8] = str(5.0)
#             line_rejoined = '\t'.join(line_fragmented)
#             f.write("%s\n" % line_rejoined)
#     # print(len(content))


