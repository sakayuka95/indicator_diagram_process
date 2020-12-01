"""
created on 2020/11/27 11:45
@author:Xia Feng
@note:1.calculate area of a diagram
      2.calculate area ratio
"""

import load
import generate_origin
import numpy as np


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


def count_area_ratio(img_name1, img_name2):
    s1 = count_area_by_name(img_name1)
    s2 = count_area_by_name(img_name2)
    return float(s1 / s2)


def count_area_by_name(img_name):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    data, dict_index = loadfile.get_pixel_data()
    index = dict_index.get(img_name)
    x, f = load.get_pixel_by_index(index, data)
    arr = generate_origin.process_pixel(x, f)
    s = count_area(arr)
    return s


if __name__ == '__main__':
    ratio = count_area_ratio('FD-1-12968.png', 'FD-1-12867.png')
    print(ratio)
