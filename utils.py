import os
import time
from pathlib import Path
import glob
import numpy as np
import sys

MAX_LABEL_LEN = 8

CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ"
print(len(CHARS))
# to utf-8
CHARS = CHARS.encode('utf-8').decode('utf-8')
print(CHARS)
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}

# South Korea city
koreaCity = {
    '서울':'A', '부산':'B', '대구':'C', '인천':'D',
    '광주':'E', '대전':'F', '울산':'G', '세종':'H',
    '경기':'I', '강원':'J', '충북':'K', '충남':'L',
    '전북':'M', '전남':'N', '경북':'O', '경남':'P',
    '제주':'Q'
}

# function to_label
# input: string
# output: list of ints
# example: "제주79바4470" -> [83, 7, 9, 45, 4, 4, 7, 0]
def to_label(text):
    text = text.encode('utf-8').decode('utf-8')
    print("len: ", len(text))
    # if first two characters are korea city
    if text[:2] in koreaCity:
        text = koreaCity[text[:2]] + text[2:]

    # to list of ints
    ints = []
    for c in text:
        print(c)
        ints.append(CHARS_DICT[c])
    return ints



# function string_to_ints
# input: string
# output: list of ints
# example: "123" -> [1, 2, 3]
def string_to_ints(text):
    ints = []
    for c in text:
        ints.append(CHARS_DICT[c])
    return ints


# function ints_to_string
# input: list of ints
# output: string
# example: [1, 2, 3] -> "123"
def ints_to_string(ints):
    string = ""
    for i in ints:
        string += DECODE_DICT[i]
    return string


# test_txt = "제주79바4470"
# print(to_label(test_txt))


if __name__ == "__main__":
    f_path = "data/*.jpg"
    for i in glob.glob(f_path):
        f_base = os.path.basename(i)
        f_name = os.path.splitext(f_base)[0].split("_")[0]
        print(f_name)
        label = to_label(f_name)
        print(label)
