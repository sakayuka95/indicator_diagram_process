"""
created on 2020/11/27 11:45
@author:Xia Feng
@note:1.draw pixel diagram by name
      2.calculate area of a diagram
      3.calculate area ratio
"""

import load
import numpy as np
import cv2 as cv


def draw_pixel(img, x, f):
    for cnt in range(len(x) - 1):
        cv.line(img, (x[cnt], f[cnt]), (x[cnt + 1], f[cnt + 1]), 0, thickness=1)
    cv.line(img, (x[-1], f[-1]), (x[0], f[0]), 0, thickness=1)


def process_pixel(x_new, f_new):
    arr = np.full((256, 256), 255, np.uint8)
    draw_pixel(arr, x_new, f_new)
    img2 = np.flip(arr, axis=0)
    return img2


def count_area(img_arr):
    list_up = []
    list_down = []
    for col in range(len(img_arr)):
        tmp = img_arr[:, col]
        for row in range(len(img_arr[0])):
            if tmp[row] < 200:
                list_up.append(row)
                break
        for row in reversed(range(len(img_arr[0]))):
            if tmp[row] < 200:
                list_down.append(row)
                break
    tmp1 = np.array(list_up)
    tmp2 = np.array(list_down)
    tmp = tmp2 - tmp1
    s = tmp.sum()
    return s


def count_area_by_name(img_name):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    data, dict_index = loadfile.get_pixel_data()
    index = dict_index.get(img_name)
    x, f = load.get_pixel_by_index(index, data)
    arr = process_pixel(x, f)
    s = count_area(arr)
    return s


def count_area_ratio(img_name1, img_name2):
    s1 = count_area_by_name(img_name1)
    s2 = count_area_by_name(img_name2)
    return float(s1 / s2)


if __name__ == '__main__':
    ratio = count_area_ratio('FD-1-12968.png', 'FD-1-12867.png')
    print(ratio)
