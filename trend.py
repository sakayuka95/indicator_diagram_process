"""
created on 2020/11/30 11:10
@author:Xia Feng yuka
@note:trend analysis:
      1.generate trend video from image list
      2.draw in one image
"""

import os
import os.path as osp
import cv2 as cv
import load
import numpy as np
import generate_origin


def cmp(x):
    return int(x.split('-')[-1][:-4])


def show_pic(path, interval):
    source_path = path
    jpg_set = os.listdir(source_path)[0:-1:interval]
    jpg_set = sorted(jpg_set, key=cmp)
    cv.waitKey(1000)
    for img_name in jpg_set:
        img_arr = cv.imread(osp.join(source_path, img_name), cv.IMREAD_GRAYSCALE)
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
        frame = cv.imread(osp.join(source_path, img_name))
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


# deprecated
def draw_in_one(temp, pf, save_path):
    img = 255 * np.ones((256, 256, 3), np.uint8)
    for index in temp:
        x, f = load.get_pixel_by_index(index, pf)
        generate_origin.draw_line(img, x, f, (255, 0, 0))
        # print(index)
    img2 = cv.flip(img, 0, dst=None)
    cv.imwrite(save_path, img2)


# deprecated
def process_in_one_image(equipment, save_path, interval):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    idx = 0
    index_list = []
    for name in pf['image_name']:
        if name.find(equipment) != -1:
            index = dict_index.get(name)
            index_list.append(index)
    while idx + interval < len(index_list):
        temp = index_list[idx:idx + interval]
        path = osp.join(save_path, equipment + '-' + str(idx) + '.png')
        draw_in_one(temp, pf, path)
        idx += interval
    temp = index_list[idx:len(index_list) - 1]
    path = osp.join(save_path, equipment + '-' + str(idx) + '.png')
    draw_in_one(temp, pf, path)
    idx += interval


if __name__ == '__main__':
    # test()
    # show_pic('5', 10)
    generate_video('test111.mp4', '5/', 10)
    # process_in_one_image('FD-5', 'D:\\pythonProject\\history', 10000)
