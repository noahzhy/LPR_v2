import os
import glob
from shutil import copyfile


# list all img files in the dir
def list_all_files(root, exts=['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG']):
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root, '*.{}'.format(ext))))
    return files


# move files from src to dst
def move_files(src, dst):
    files = list_all_files(src)
    # split 0.1 for test
    files = files[:int(len(files) * 0.1)]
    for file in files:
        # copy
        # copyfile(file, os.path.join(dst, os.path.basename(file)))
        # move
        os.rename(file, os.path.join(dst, os.path.basename(file)))


# file name to lowercase
def file_name_to_lowercase(root):
    files = list_all_files(root)
    for file in files:
        os.rename(file, os.path.join(root, os.path.basename(file).lower()))


# main
if __name__ == '__main__':
    # move files from src to dst
    move_files('train', 'test')
    # file_name_to_lowercase(root='d0')