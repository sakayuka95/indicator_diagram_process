"""
created on 2020/10/28 12:10
@author:yuka
@note:load data from different ways:
      1.a csv or excel data file -> origin data
      2.a png image file -> pixel
      3.a csv hash map -> pixel(origin data after normalization)
"""

import pandas as pd
import cv2 as cv
import xlrd
import os
import os.path as osp


def get_pixel_by_index(index, data):
    x = data['x']
    f = data['f']
    x1 = list(map(eval, x.iloc[index].split(',')))
    f1 = list(map(eval, f.iloc[index].split(',')))
    return x1, f1


def get_str_value(value, param):
    string = ''
    for i in range(1, len(value)):
        string += value[i].split(',')[param]
        if i != len(value)-1:
            string += ' '
    return string


def parse_excel():
    data = xlrd.open_workbook('D:\\pythonProject\\data\\test\\大港油田示功图测试数据.xls')
    # name_set = set()
    table = data.sheets()[0]
    row = table.nrows
    for r in range(1, row):
        device_name = table.cell(r, 2).value
        data_value = table.cell(r, 22).value
        if data_value == '':
            continue
        device_path = osp.join('D:\\pythonProject\\txtdata', device_name)
        # if device_name not in name_set:
        #     name_set.add(device_name)
        if not os.path.exists(device_path):
            os.makedirs(device_path)
        file_path = osp.join('D:\\pythonProject\\txtdata', device_name, device_name+'_'+str(r)+'.txt')
        print(file_path)
        data_value = table.cell(r, 22).value
        data_value = data_value.replace('\r', '')
        value = data_value.split('\n')
        length = len(value)
        x = get_str_value(value[1:length-1], 1)
        print(x)
        f = get_str_value(value[1:length-1], 0)
        print(f)
        with open(file_path, 'w') as fw:
            fw.write(x + '\n')
            fw.write(f + '\n')


class LoadData:

    def __init__(self, path):
        self.filepath = path

    def read_excel(self):
        return pd.read_excel(self.filepath).iloc[:, [8, 11]].values

    def read_csv(self):
        return pd.read_csv(self.filepath, usecols=['位移', '载荷'], encoding='ANSI')

    def get_data_from_image(self):
        img_arr_origin = cv.imread(self.filepath, cv.IMREAD_GRAYSCALE)
        # 绘制时翻转回来
        img_arr = cv.flip(img_arr_origin, 0, dst=None)
        span = list(range(256))[:-1:1]
        list_up = []
        list_down = []
        cnt = 0
        for col in span:
            tmp = img_arr[:, col]
            # not blank
            if (tmp == 255).any != 0:
                if cnt % 2 == 0:
                    for row in range(256):
                        if tmp[row] < 200:
                            list_up.append((col, row))
                            break
                else:
                    for row in reversed(range(256)):
                        if tmp[row] < 200:
                            list_down.append((col, row))
                            break
            cnt += 1
        list_down.reverse()
        list_data = list_up + list_down
        return list_data

    def get_pixel_data(self):
        data = pd.read_csv(self.filepath)
        dict_origin = data['image_name'].to_dict()
        dict_index = {value: key for key, value in dict_origin.items()}
        return data, dict_index
