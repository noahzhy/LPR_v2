import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# show korean in matplotlib
plt.rcParams["font.sans-serif"]=["Batang"]



# list all file in dir_path
def list_all_files(dir_path):
    letters = {}

    files = glob.glob(dir_path + '/*.jpg')
    # list all files name
    files_name = []
    for file in files:
        f_name = os.path.basename(file)
        # split file name via '.'
        f_name = f_name.split('.')[0].split('_')[0]
        files_name.append(f_name)

        # 统计每个字符出现的次数
        for letter in f_name:
            if letter not in letters.keys():
                letters[letter] = 1
            else:
                letters[letter] += 1

    # sort by value
    letters = dict(sorted(letters.items(), key=lambda item: item[1], reverse=True))
    # remove space dict
    letters.pop(' ')
    return letters


# main
if __name__ == '__main__':
    letters = list_all_files('data')
    plt.figure(figsize=(15, 5))
    # show it in bar with value
    plt.bar(range(len(letters)), list(letters.values()), tick_label=list(letters.keys()))
    plt.title('Korean License Plate Letters')
    plt.xlabel('Letters')
    plt.ylabel('Count')
    plt.savefig('letters.png')
    plt.show()
