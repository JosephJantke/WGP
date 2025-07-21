import glob
import os
from pathlib import Path
import re
import shutil

# os.chdir("E:/joseph/model_0/inference/snippets_for_raven/old/no_wgp_calls/snippets_14_SCR31")
# folder = "E:/joseph\model_0/inference/snippets_for_raven\old/no_wgp_calls/snippets_14_SCR31/**/*.wav"
#
# for file in glob.glob(folder, recursive=True):
#     join_path = os.path.join(folder, file)
#     # print(file)
#     original = file
#     name = file[82:103]
#     # print(name)
#     name_path = file[64:81]
#     print(name_path)
#     destination = file[:82] + name_path + "_" + name + "_0.wav"
#     print(destination)
#     shutil.copy(original, destination)


#replacing all whitespace with underscore in a directory
os.chdir("D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_calls")
folder = "D:/PhD/WGP_model/from_toshiba_dbca_examples/CANP_monitoring_ARU_calls/**/*.wav"

for file in glob.glob(folder, recursive=True):
    join_path = os.path.join(folder, file)
    original = file
    print(original)
    destination = original.replace(' ', '_')
    print(destination)
    shutil.copy(original, destination)
    os.remove(original)