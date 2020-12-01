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


class LoadData:

    def __init__(self, path):
        self.filepath = path

    def read_excel(self):
        return pd.read_excel(self.filepath).iloc[:, [8, 11]].values

    def read_csv(self):
        return pd.read_csv(self.filepath, usecols=['位移', '载荷'], encoding='ANSI')

    def get_data_from_image(self):
        img_arr = cv.imread(self.filepath, cv.IMREAD_GRAYSCALE)
        span = list(range(256))[:-1:1]
        list_up = []
        list_down = []
        cnt = 0
        for col in span:
            tmp = img_arr[:, col]
            if (tmp == 255).any != 0:
                if cnt % 2 == 0:
                    for row in range(256):
                        if tmp[row] < 50:
                            list_up.append((col, row))
                            break
                else:
                    for row in reversed(range(256)):
                        if tmp[row] < 50:
                            list_down.append((col, row))
                            break
            cnt += 1
        list_down.reverse()
        list_data = list_up + list_down
        return list_data

    def get_pixel_by_name(self, img_name):
        data = pd.read_csv(self.filepath)
        dict_origin = data['image_name'].to_dict()
        dict_index = {value: key for key, value in dict_origin.items()}
        index = dict_index.get(img_name)
        x = data['x']
        f = data['f']
        x1 = list(map(eval, x.iloc[index].split(',')))
        f1 = list(map(eval, f.iloc[index].split(',')))
        return x1, f1
