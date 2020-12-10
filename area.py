"""
created on 2020/11/27 11:45
@author:Shar
@note:1.draw pixel diagram by name
      2.calculate area of a diagram
      3.calculate area ratio
"""

import load
import numpy as np
import cv2 as cv
import math
import os
import pandas as pd
import util


def draw_pixel(img, x, f):
    for tmp in range(len(x) - 1):
        cv.line(img, (x[tmp], f[tmp]), (x[tmp + 1], f[tmp + 1]), 0, thickness=1)
    cv.line(img, (x[-1], f[-1]), (x[0], f[0]), 0, thickness=1)


def process_pixel(x_new, f_new):
    background_arr = np.full((256, 256), 255, np.uint8)
    draw_pixel(background_arr, x_new, f_new)
    img2 = np.flip(background_arr, axis=0)
    return img2


# 简单归一化，x和f均是根据全局变量的最大值进行放缩，而不是局部最大值
def normalization(x, data_max):
    if x_max.any() == 0:
        return data_max, x
    alpha = 253 / data_max
    x = x * alpha
    return data_max, x


def getArea(org_data):
    # data是最原始的示功图载荷位移数据
    num = 0
    up_index = 0
    down_index = 0
    s = 0
    # 此处默认示功图数据点数为200，如果为144点数据需要修改常量198为142
    while num < 198:
        if num % 2 == 0:
            p1 = org_data[up_index]
            p2 = org_data[up_index + 1]
            p3 = org_data[199 - down_index]
            a = float(math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1])))
            b = float(math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1])))
            c = float(math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))
            half_of_perimeter = (a + b + c) / 2
            s = s + (half_of_perimeter * (half_of_perimeter - a) * (half_of_perimeter - b) * (
                        half_of_perimeter - c)) ** 0.5
            up_index += 1
        else:
            p1 = org_data[199 - down_index]
            p2 = org_data[199 - down_index - 1]
            p3 = org_data[up_index]
            a = float(math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1])))
            b = float(math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1])))
            c = float(math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))
            half_of_perimeter = (a + b + c) / 2
            s = s + (half_of_perimeter * (half_of_perimeter - a) * (half_of_perimeter - b) * (
                        half_of_perimeter - c)) ** 0.5
            down_index += 1
        num += 1
    return s


class Img2Area:
    def __init__(self):
        self.dict = {
            'FD-1': (5.272, 119.84),
            'FD-2': (5.891, 65.64),
            'FD-3': (6.076, 69.13),
            'FD-4': (5.109, 61.06),
            'FD-5': (5.421, 47.18),
            'FD-6': (5.981, 87.91),
            'FD-7': (7.132, 61.97),
            'FD-8': (5.998, 78.41),
        }

    def count_area(self, img_name, img_arr):
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
        data_max = self.dict.get(img_name[0:4])
        mul = (253 / data_max[0]) * (253 / data_max[1])
        return s / mul

    def count_area_ratio(self, img_name1, img_name2):
        s_1 = self.count_area_by_name(img_name1)
        s_2 = self.count_area_by_name(img_name2)
        return float(s_1 / s_2)

    def count_area_by_name(self, img_name):
        loadfile = load.LoadData('D:\\data\\name_point\\image_data_map.csv')
        pixel_data, dict_index = loadfile.get_pixel_data()
        index = dict_index.get(img_name)
        x, f = load.get_pixel_by_index(index, data)
        img_arr = process_pixel(x, f)
        s = self.count_area(img_name, img_arr)
        return s


if __name__ == '__main__':
    # ratio = img2area().count_area_ratio('FD-1-12968.png', 'FD-1-12867.png')
    # print(ratio)
    root = 'D:\\data'
    data_path = os.path.join(root, 'orgData')
    listdir = os.listdir(data_path)
    for loc in listdir:
        if loc != 'FD-2.csv':
            continue
        data = pd.read_csv(os.path.join(data_path, loc), encoding='GBK', usecols=['位移', '载荷'])
        data = data.values
        data = np.array(data)
        x_max = np.max(data, axis=0)
        base = 200
        cnt = data.shape[0] // base
        for idx in range(cnt):
            # 选取示功图的200个点
            data_idx = data[idx * base:(idx + 1) * base, :]
            if util.zero_data(data_idx[0], data_idx[1]):
                continue

            # 原始的200个点算功图面积
            s1 = getArea(data_idx)
            print('根据映射关系回推的示功图近似面积为：' + str(s1))
            # 等比列放大归一化
            arr_max, data_idx_new = normalization(data_idx, x_max)
            if arr_max.any() == 0:
                continue
            data_tmp = np.array(data_idx_new, dtype=np.int)

            # 填充256*256像素点
            arr = np.full((256, 256), 255, np.uint8)
            data_list = [tuple(x) for x in data_tmp]

            # 可视化2
            for cnt in range(199):
                cv.line(arr, data_list[cnt], data_list[cnt + 1], 0, thickness=1)
            cv.line(arr, data_list[199], data_list[0], 0, thickness=1)
            cv.imshow('img', np.flip(arr, axis=0))
            cv.waitKey()

            # 计算面积
            name = loc.replace('.csv', '_' + str(idx) + '.jpg')
            s2 = Img2Area().count_area(name, arr)
            print('在原始数据中近似求解的面积为：' + str(s2))

            # 保存文件
            dir_name = loc.split('.')[0]
            save_path = os.path.join(root, 'data2img', dir_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv.imwrite(os.path.join(save_path, name), np.flip(arr, axis=0))
