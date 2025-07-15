import glob
import os
from pathlib import Path
import re
import shutil

#FOR CREATING TXT FOR TP FOR "tp" directory

os.chdir("E:/joseph/model_3/train/tp")
folder = "E:/joseph/model_3/train/tp/**/*.wav"

with open("E:/joseph/model_3/train/tp_train.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[27:])
        f.write("1," + name + "\n")

#FOR CREATING TXT FOR FP FOR "fp" directory

os.chdir("E:/joseph/model_3/train/fp")
folder = "E:/joseph/model_3/train/fp/**/*.wav"

with open("E:/joseph/model_3/train/fp_train.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[27:])
        f.write("0," + name + "\n")

#todo

#FOR CREATING TXT FOR TP for "train.txt"

os.chdir("E:/joseph/model_4/train")
folder = "E:/joseph/model_4/train/tp/**/*.wav"

with open("E:/joseph/model_4/train/tp_train_train.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[27:])
        f.write("1,tp/" + name + "\n")

#FOR CREATING TXT FOR FP for "train.txt"

os.chdir("E:/joseph/model_4/train")
folder = "E:/joseph/model_4/train/fp/**/*.wav"

with open("E:/joseph/model_4/train/fp_train_train.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[27:])
        f.write("0,fp/" + name + "\n")

#todo
#FOR CREATING TXT FOR TP for "valid.txt"

os.chdir("E:/joseph/model_1/train/tp_valid")
folder = "E:/joseph/model_1/train/tp_valid/**/*.wav"

with open("E:/joseph/model_1/train/tp_valid.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[33:])
        # print(name)
        f.write("1,tp_valid/" + name + "\n")

#FOR CREATING TXT FOR FP for "valid.txt"

os.chdir("E:/joseph/model_1/train/fp_valid")
folder = "E:/joseph/model_1/train/fp_valid/**/*.wav"

with open("E:/joseph/model_1/train/fp_valid.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[33:])
        f.write("0,fp_valid/" + name + "\n")




#TODO MAKING TEXT FILES FOR "test.txt"
os.chdir("E:/joseph/model_1/train/tp_test")
folder = "E:/joseph/model_1/train/tp_test/**/*.wav"

with open("E:/joseph/model_1/train/tp_test.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[32:])
        # print(name)
        f.write("1,tp_test/" + name + "\n")

# FOR CREATING TXT FOR FP for "valid.txt"

os.chdir("E:/joseph/model_1/train/fp_test")
folder = "E:/joseph/model_1/train/fp_test/**/*.wav"

with open("E:/joseph/model_1/train/fp_test.txt", 'x') as f:
    for file in glob.glob(folder, recursive=True):
        join_path = os.path.join(folder, file)
        name = (file[32:])
        print(name)
        f.write("0,fp_test/" + name + "\n")

