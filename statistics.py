"""
created on 2020/12/1 11:10
@author:yuka
@note:statistics util
"""

import numpy as np
import os
import os.path as osp
import load
import util


def get_csv_max(filepath):
    fileset = os.listdir(filepath)
    for fileName in fileset:
        loadfile = load.LoadData(osp.join(filepath, fileName))
        file = loadfile.read_csv()
        pre = osp.splitext(osp.basename(fileName))[0]
        data = np.array(file)
        x = data[:, 0]
        f = data[:, 1]
        print(pre + ' : max_x= ' + str(max(x)) + ' : max_f= ' + str(max(f)))


def get_excel_max(filepath):
    fileset = os.listdir(filepath)
    for fileName in fileset:
        loadfile = load.LoadData(osp.join(filepath, fileName))
        data = loadfile.read_csv()
        pre = osp.splitext(osp.basename(fileName))[0]
        f_max = 0
        x_max = 0
        for cnt in range(len(data)):
            if util.nan_data(data[cnt, 0]) | util.nan_data(data[cnt, 1]):
                continue
            x = data[cnt, 0].split(',')
            f = data[cnt, 1].split(',')
            if len(x) != len(f):
                continue
            x_array = np.array(x, dtype=float)
            f_array = np.array(f, dtype=float)
            f_max = max(f_max, max(f_array))
            x_max = max(x_max, max(x_array))
        print(pre + " max_f: " + str(f_max) + " max_x: " + str(x_max))


# deprecated
def get_2classify_num(image_name_set):
    minus_num = 0
    plus_num = 0
    for image_name in image_name_set:
        if image_name.find('tm') != -1:
            minus_num += 1
        elif image_name.find('tp') != -1 or image_name.find('vs') != -1:
            plus_num += 1
    print('minus: ' + str(minus_num) + ', plus: ' + str(plus_num))
    return minus_num, plus_num


# deprecated
def get_2classify_total_num_test(filepath):
    fileset = os.listdir(filepath)
    total_num = 0
    minus_num = 0
    plus_num = 0
    for image_path in fileset:
        image_name_set = os.listdir(osp.join(filepath,image_path))
        num = len(image_name_set)
        total_num += num
        print(image_path + ' has ' + str(num) + ' images ')
        minus_temp, plus_temp = get_2classify_num(image_name_set)
        minus_num += minus_temp
        plus_num += plus_temp
    print('total num is: ' + str(total_num))
    print('minus num is: ' + str(minus_num))
    print('plus num is: ' + str(plus_num))


def get_total_num(filepath):
    fileset = os.listdir(filepath)
    total_num = 0
    for image_path in fileset:
        image_name_set = os.listdir(osp.join(filepath,image_path))
        num = len(image_name_set)
        total_num += num
        print(image_path + ' has ' + str(num) + ' images ')
    print('total num is: ' + str(total_num))


def get_each_label_count(path):
    image_path_set = os.listdir(path)
    print(path + ' has ' + str(len(image_path_set)) + ' labels ')
    for label in image_path_set:
        image_name_set = os.listdir(path + '\\' + label)
        label_num = len(image_name_set)
        print(label + ' has ' + str(label_num) + ' images ')


if __name__ == '__main__':
    get_csv_max('D:\\pythonProject\\data\\csv')
    get_excel_max('D:\\pythonProject\\data\\excel')
