from hangul_utils import *
import numpy as np

word = """
가거고구
나너노누
다더도두
라러로루
마머모무
바버보부
사서소수
아어오우
자저조주
하허호배
"""
# convert char to list

_list = []

for char in word:
    # pass \n
    if char == '\n':
        continue
    _list.append(char)

print(_list)
# convert list to numpy array
arr = np.array(_list)
# convert numpy array to shape (10, 4)
arr = arr.reshape(10, 4)
# exchange 2nd column and 4th column
arr[:, [1, 2]] = arr[:, [2, 1]]

shifts = np.arange(arr.shape[1])
for i, shift in enumerate(shifts):
    arr[:, i] = np.roll(arr[:, i], shift*2)



print(arr)
# reshpae to 1d array，and convert to list
_list = arr.reshape(-1).tolist()
# convert list to string
word = ''.join(_list)
print(word)

word = """
0가A조a서B무b
1나C호c어D부d
2다E고e저F수f
3라G노g허H우h
4마I도i거J주j
5바K로k너L배l
6사M모m더N구n
7아O보o러P누p
8자Q소q머두하
9오버루
"""

# remove \n and space
word = word.replace('\n', '')
print(word)
print(len(word))



# word = """0A8가9B호C저D우E나F고G허H주I다J노K거L배M라N도O너P구Q마a로b더c누d바e모f러g두h사i보j머k루l아m소n버o무p자q오1서2부3하4조5어6수7 """
print(len(word))
