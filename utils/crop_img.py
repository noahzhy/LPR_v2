# crop given area from frame of video
import os
import sys
import glob

import cv2
import numpy as np

from PIL import *


# function to crop given area from frame of video
def crop_img(img, x, y, w, h):
    return img[y:y+h, x:x+w]


# split the video into frames each 1 second
def split_video(video_path, output_dir):
    # create output dir if not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # open video
    cap = cv2.VideoCapture(video_path)
    # get total frame count
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # get fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get total seconds
    total_seconds = total_frames / fps
    # save frame each 1 second
    for i in range(int(total_seconds)):
        # get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * fps))
        ret, frame = cap.read()
        # save frame with video name
        cv2.imwrite(os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '_' + str(i) + '.jpg'), frame)
    # release video
    cap.release()


# main
if __name__ == '__main__':
    # # list all videos in given dir
    # videos_root = r"D:\road_test_video"
    # videos = glob.glob(os.path.join(videos_root, '*.mp4'))
    # # split each video into frames
    # for video in videos:
    #     split_video(video, os.path.join(videos_root, 'frames'))

    # list all images in given dir
    images_root = r"D:\road_test_video\frames"
    images = glob.glob(os.path.join(images_root, '*.jpg'))
    # crop each image
    for image in images:
        img = cv2.imread(image)
        # crop given area
        # x1, y1, x2, y2 = 427, 562, 851, 718
        img = crop_img(img, 427, 562, 851 - 427, 718 - 562)
        # save
        cv2.imwrite(image, img)

