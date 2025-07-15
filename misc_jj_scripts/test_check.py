import csv
import random
import glob
import shutil
import os
import pandas as pd
from pandas import *

#read total.csv into a list
filename = 'csv/total.csv'
column_name = 'wavname'  # Replace with the name of the column you want to read

# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

# Print the values of the specified column
wavlist = df[column_name].tolist()







file_path = "C:/Users/josep/OneDrive/Documents/University/PhD/recogniser/total.csv"

with open("C:/Users/josep/OneDrive/Documents/University/PhD/recogniser/total.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimeter=',')
    line_count = 0
    for row in csv_reader:
        print(row)

df = pd.read_csv(file_path)
    print(df)



for csv in csv_files:
    file_path = os.path.join(folder_path, csv)
    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(file_path, skiprows=10)
        print(df)
        #df = df.drop(index=1)
        df_list.append(df)
        print(df_list)
    except UnicodeDecodeError: