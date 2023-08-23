import os
import glob
import numpy as np
from PIL import Image

# South Korea city
koreaCity = {
    '서울': 'A', '부산': 'B', '대구': 'C', '인천': 'D',
    '광주': 'E', '대전': 'F', '울산': 'G', '세종': 'H',
    '경기': 'I', '강원': 'J', '충북': 'K', '충남': 'L',
    '전북': 'M', '전남': 'N', '경북': 'O', '경남': 'P',
    '제주': 'Q'
}

# list all img files in the dir
def list_all_files(root, exts=['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, '*.{}'.format(ext))))
    return files


# main
if __name__ == '__main__':
    path = r'aihub\double\yellow'
    files = list_all_files(path)
    for i, file in enumerate(files):
        # remove space
        new_name = os.path.basename(file).replace(' ', '')
        # add ' ' at 3rd char
        new_name = new_name[:3] + ' ' + new_name[3:]
        print('rename {} to {}'.format(os.path.basename(file), new_name))
        os.rename(file, os.path.join(path, new_name))
        # break

        # # if first two chars are both korean letters
        # if os.path.basename(file)[:2] in koreaCity.keys():
        #     # korean city
        #     korea_city_letter = koreaCity[os.path.basename(file)[:2]].lower()
        #     new_name = korea_city_letter + os.path.basename(file)[2:]
        #     print('rename {} to {}'.format(os.path.basename(file), new_name))
        #     os.rename(file, os.path.join(path, new_name))
        #     # print('rename {} to {}'.format(os.path.basename(file), new_name))

