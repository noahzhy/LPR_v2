import os
import random
import numpy as np


MAX_LABEL_LEN = 8

# CHARS = "0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주하허호바사아자배abcdefghijklmnopqABCDEFGHIJKLMNOPQ "
CHARS = """ 0가A조a서B무b1나C호c어D부d2다E고e저F수f3라G노g허H우h4마I도i거J주j5바K로k너L배l6사M모m더N구n7아O보o러P누p8자Q소q머두하9오버루"""
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
DECODE_DICT = {i: char for i, char in enumerate(CHARS)}
# South Korea city
koreaCity = {
    '서울': 'A', '부산': 'B', '대구': 'C', '인천': 'D',
    '광주': 'E', '대전': 'F', '울산': 'G', '세종': 'H',
    '경기': 'I', '강원': 'J', '충북': 'K', '충남': 'L',
    '전북': 'M', '전남': 'N', '경북': 'O', '경남': 'P',
    '제주': 'Q'
}


def str2list(string):
    res = []
    # if first two chars are both korean letters
    if string[:2] in koreaCity.keys():
        # korean city
        korea_city_letter = koreaCity[string[:2]]
        string = korea_city_letter + string[2:]

    for char in string:
        res.append(CHARS_DICT[char])
    return res


def decoder(file_path):
    lp = ""
    # get file base name from file path
    f_base = os.path.basename(file_path).split('.')[0]

    # split via '_' to a list
    f_base = f_base.split('_')
    for i in f_base:
        if len(lp+i) > MAX_LABEL_LEN+1:
            break
        lp += i

    return str2list(lp)


if __name__ == "__main__":
    f_path = "제주45_다1234_5678345.jpg"
    print(decoder(f_path))
    f_path = "Q45_다1234_356892.jpg"
    print(decoder(f_path))
    f_path = "제주45다_5462_20230115213242_14223844.jpg"
    print(decoder(f_path))
    f_path = "Q45다5462_20230115213242_14223844.jpg"
    print(decoder(f_path))
    f_path = "45다5462.jpg"
    print(decoder(f_path))