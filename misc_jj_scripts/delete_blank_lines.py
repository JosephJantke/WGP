from pathlib import Path
import glob
import os

#todo remember to specify the home and folder

home = "E:/SZ673_BlackchinnedHoneyeater/jj"
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