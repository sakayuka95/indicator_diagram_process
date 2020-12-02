import math
import os

import cv2 as cv
import numpy as np
import pandas as pd


def generateImgByName(img_name, map_csv_path='D:\\data\\name_point\\image_data_map.csv'):
    data = pd.read_csv(map_csv_path)
    dict_origin = data['image_name'].to_dict()
    dict_index = {value: key for key, value in dict_origin.items()}
    index = dict_index.get(img_name)
    x = data['x']
    f = data['f']
    x = list(map(eval, x.iloc[index].split(',')))
    f = list(map(eval, f.iloc[index].split(',')))
    # print(type(x))
    arr = np.full((256, 256), 255, np.uint8)
    for cnt in range(len(x) - 1):
        cv.line(arr, (x[cnt], f[cnt]), (x[cnt + 1], f[cnt + 1]), 0, thickness=1)
    cv.line(arr, (x[-1], f[-1]), (x[0], f[0]), 0, thickness=1)
    arr = np.flip(arr, axis=0)
    return arr


class img2area():
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

    def countArea(self, img_name, img_arr):
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
        mul = (256 / data_max[0]) * (256 / data_max[1])
        return s / mul


def normalization2(x, x_max):
    if x_max.any() == 0:
        return x_max, x
    alpha = 256 / x_max
    x = x * alpha
    return x_max, x


def zero_data(x, f):
    # case 1 : zero
    case1 = (x == 0).all() | (f == 0).all()
    # case 2 : close to zero
    case2 = abs(max(f) - min(f)) < 1
    return case1 or case2


def getArea(data):
    cnt = 0
    up_index = 0
    down_index = 0
    s = 0
    while cnt < 198:
        if cnt % 2 == 0:
            p1 = data[up_index]
            p2 = data[up_index + 1]
            p3 = data[199 - down_index]
            a = float(math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1])))
            b = float(math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1])))
            c = float(math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))
            l = (a + b + c) / 2
            s = s + (l * (l - a) * (l - b) * (l - c)) ** 0.5
            up_index += 1
        else:
            p1 = data[199 - down_index]
            p2 = data[199 - down_index - 1]
            p3 = data[up_index]
            a = float(math.sqrt((p2[0] - p3[0]) * (p2[0] - p3[0]) + (p2[1] - p3[1]) * (p2[1] - p3[1])))
            b = float(math.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1])))
            c = float(math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])))
            l = (a + b + c) / 2
            s = s + (l * (l - a) * (l - b) * (l - c)) ** 0.5
            down_index += 1
        cnt += 1
    return s


def main():
    root = 'D:\data'
    data_path = os.path.join(root, 'orgdata')
    listdir = os.listdir(data_path)
    for loc in listdir:
        if loc != 'FD-4.csv':
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
            if zero_data(data_idx[0], data_idx[1]):
                continue

            # 原始的200个点算功图面积
            s1 = getArea(data_idx)
            print('根据映射关系回推的示功图近似面积为：' + str(s1))
            # 等比列放大归一化
            arr_max, data_idx_new = normalization2(data_idx, x_max)
            if arr_max.any() == 0:
                continue
            data_tmp = np.array(data_idx_new, dtype=np.int)

            # 填充100*100像素点
            # arr = np.zeros([500, 500], np.uint8)
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
            s2 = img2area().countArea(name, arr)
            print('在原始数据中近似求解的面积为：' + str(s2))

            # 保存文件
            dir_name = loc.split('.')[0]
            save_path = os.path.join(root, 'data2img', dir_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv.imwrite(os.path.join(save_path, name), np.flip(arr, axis=0))


if __name__ == '__main__':
    # ratio = img2area().count_area_ratio('FD-1-12968.png', 'FD-1-12867.png')
    # print(ratio)
    main()
