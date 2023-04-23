import os
from PIL import Image, ImageDraw, ImageFont

import matplotlib
from matplotlib import pyplot as plt


TARGET_WIDTH = 96
TARGET_HEIGHT = 48

# print radio
print('Radio: ', TARGET_WIDTH / TARGET_HEIGHT)


# open image via path and keep Green channel only
def open_image(path, channel='G'):
    img = Image.open(path)
    img = img.convert('RGB')
    r, g, b = img.split()
    if channel == 'R':
        return r
    elif channel == 'G':
        return g
    elif channel == 'B':
        return b
    else:
        return img.convert('L')


# create a new image with black background
def create_image(raw_img):
    img = Image.new('L', (TARGET_WIDTH, TARGET_HEIGHT), (0))
    # resize the raw image to fit the target image width, and keep the ratio
    raw_img = raw_img.resize(
        (TARGET_WIDTH, int(raw_img.size[1] * TARGET_WIDTH / raw_img.size[0])),
        Image.ANTIALIAS
    )
    # paste the raw image to the target image at top
    img.paste(raw_img, (0, 0))
    return img


if __name__ == '__main__':
    # open the raw image via path
    # path = 'data/88거9742_1625465514.jpg'       # black-white
    # path = 'data/79버1012_1625465514.jpg'       # green
    path = 'data/제주79바4470_1625339435.jpg'    # yellow
    # path = 'data/55구1601_1625339435.jpg'       # blue
    # print its radio
    print('Radio: ', Image.open(path).size[0] / Image.open(path).size[1])

    # four imgs to show
    fig = plt.figure(figsize=(8, 5))

    for i, color in zip(range(1, 5), ['R', 'G', 'B', 'L']):
        fig.add_subplot(2, 2, i)
        # title
        plt.title(color)
        img = open_image(path, channel=color)

        # create a new image
        img = create_image(img)
        plt.imshow(img, cmap='gray')

    # # show
    # plt.imshow(img, cmap='gray')
    # plt.show()
    # save as png
    plt.savefig('result.png')
