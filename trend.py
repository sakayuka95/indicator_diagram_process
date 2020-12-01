"""
created on 2020/11/30 11:10
@author:Xia Feng yuka
@note:generate trend video from image list
"""

import os
import cv2 as cv


def cmp(x):
    return int(x.split('-')[-1][:-4])


def show_pic(path, interval):
    source_path = path
    jpg_set = os.listdir(source_path)[0:-1:interval]
    jpg_set = sorted(jpg_set, key=cmp)
    cv.waitKey(1000)
    for img_name in jpg_set:
        img_arr = cv.imread(os.path.join(source_path, img_name), cv.IMREAD_GRAYSCALE)
        cv.imshow('img', img_arr)
        cv.waitKey(200)
    # cv.imshow('img', img_arr)
    # cv.waitKey()


def generate_video(video_dir, im_dir, interval):
    source_path = im_dir
    jpg_set = os.listdir(source_path)[0:-1:interval]
    jpg_set = sorted(jpg_set, key=cmp)
    img = cv.imread("FD-5-4339.png")
    size = img.shape[:2]
    print(size)
    forcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_dir, forcc, 10, (size[1], size[0]))
    for img_name in jpg_set:
        frame = cv.imread(os.path.join(source_path, img_name))
        out.write(frame)


def test():
    img = cv.imread("FD-5-4339.png")
    size = img.shape[:2]
    forcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter("3.mp4", forcc, 10, (size[1], size[0]))
    # forcc = cv.VideoWriter_fourcc(*'XVID')
    # out = cv.VideoWriter("3.avi", forcc, 10, (size[1], size[0]))
    out.write(img)
    print(size)


if __name__ == '__main__':
    # test()
    # show_pic('5', 10)
    generate_video('test111.mp4', '5/', 10)
