"""
created on 2020/10/26 12:10
@author:yuka
@note:generate origin indicator diagram
      1.get data from a excel or csv file
      2.normalization
      3.draw indicator diagram
      4.save image as a png file
      5.generate origin map:png file name; x pixel; f pixel
        5.1 from image
        5.2 from pixel map
        5.3 from origin
      6.data augmentation
        6.1 from image
        6.2 from pixel map
        6.3 from origin(deprecated)
      7.excel -> file(one device):txt files
"""

import cv2 as cv
import numpy as np
import util
import load
import random
import matplotlib.pyplot as plt
import os
import os.path as osp
import pandas as pd
import xlrd


def get_str_value(value, param):
    string = ''
    for i in range(1, len(value)):
        string += value[i].split(',')[param]
        if i != len(value) - 1:
            string += ' '
    return string


def parse_excel(file_path, save_path):
    data = xlrd.open_workbook(file_path)
    # name_set = set()
    table = data.sheets()[0]
    row = table.nrows
    for r in range(1, row):
        device_name = table.cell(r, 2).value
        data_value = table.cell(r, 22).value
        if data_value == '':
            continue
        device_path = osp.join(save_path, device_name)
        # if device_name not in name_set:
        #     name_set.add(device_name)
        if not os.path.exists(device_path):
            os.makedirs(device_path)
        file_path = osp.join(save_path, device_name, device_name + '_' + str(r) + '.txt')
        print(file_path)
        data_value = table.cell(r, 22).value
        data_value = data_value.replace('\r', '')
        value = data_value.split('\n')
        length = len(value)
        x = get_str_value(value[1:length - 1], 1)
        print(x)
        f = get_str_value(value[1:length - 1], 0)
        print(f)
        with open(file_path, 'w') as fw:
            fw.write(x + '\n')
            fw.write(f + '\n')


def generate_origin(filepath, base_path):
    fileset = os.listdir(filepath)
    for fileName in fileset:
        file = Generate(osp.join(filepath, fileName))
        if fileName.find('.xlsx') != -1:
            file.generate_excel(base_path)
        elif fileName.find('.csv') != -1:
            file.generate_csv(base_path)
            # file.generate_csv_with_axis(base_path)
        elif fileName.find('.png') != -1:
            file.generate_image(base_path)
            # file.generate_csv_with_axis(base_path)


def generate_origin_map(filepath, save_path):
    content_list = []
    fileset = os.listdir(filepath)
    for fileName in fileset:
        file = Generate(osp.join(filepath, fileName))
        if fileName.find('.xlsx') != -1:
            file.generate_excel_map(content_list)
        elif fileName.find('.csv') != -1:
            file.generate_csv_map(content_list)
        elif osp.isdir(osp.join(filepath, fileName)):
            file.generate_image_map(content_list)
    pf = pd.DataFrame(content_list, columns=["image_name", "x", "f"])
    pf.to_csv(save_path)


def generate_enhance(origin_path):
    image_path_set = os.listdir(origin_path)
    print(origin_path + ' has ' + str(len(image_path_set)) + ' labels ')
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    for label in image_path_set:
        image_path = osp.join(origin_path, label)
        image_name_set = os.listdir(image_path)
        label_num = len(image_name_set)
        print(label + ' has ' + str(label_num) + ' images ')
        if label_num < 5300:
            print(label + ' needs to be enhanced')
            data_augmentation_from_map(image_name_set, dict_index, pf, image_path)
            # data_augmentation_from_origin(image_name_set, image_path)


def data_augmentation_from_image(img_path, base_num, base_path):
    image_set = os.listdir(img_path)
    for image in image_set:
        loadfile = load.LoadData(osp.join(img_path, image))
        image_data = loadfile.get_data_from_image()
        pre = image.replace('.png', '')
        num = int(base_num + random.randint(-700, 800))
        for k in range(num):
            temp_data = image_data
            temp_list = []
            for data in temp_data:
                temp = int(data[1] + random.randint(0, 3))
                if random.randint(0, 1) <= 0.45:
                    temp_list.append((data[0], temp))
                else:
                    temp_list.append(data)
            save_path = osp.join(base_path, pre + '-' + str(k) + '.png')
            process_from_image(temp_list, save_path)


def data_augmentation_from_map(image_name_set, dict_index, pf, path):
    # order of magnitude
    label_num = len(image_name_set)
    num = int((4500 + random.randint(-700, 800)) / label_num)
    # get data source
    for image_name in image_name_set:
        index_image = dict_index.get(image_name)
        if index_image is None:
            continue
        x, f = load.get_pixel_by_index(index_image, pf)
        pre = image_name.replace('.png', '')
        for k in range(num):
            length = len(f)
            f_add = np.empty([length, 1], dtype=int)
            for i in range(length):
                temp = int(f[i] + random.randint(0, 3))
                f_add[i] = temp if random.randint(0, 1) <= 0.45 and temp <= max(f) else f[i]
            add_path = osp.join(path, pre + '-' + str(k) + '.png')
            process(x, f_add, add_path)


# deprecated
def augmentation_csv(filepath, data_set, num, base_path):
    loadfile = load.LoadData(filepath)
    file = loadfile.read_csv()
    maxlength = len(file)
    pre = osp.splitext(osp.basename(filepath))[0]
    f_max = max(np.array(file)[:, 1])
    print(pre + ' start, max_f = ' + str(f_max))
    # split
    cnt = 0
    idx = 0
    while maxlength - 200 * cnt > 0:
        # while cnt < 5:
        temp = file[idx:idx + 200]
        data = np.array(temp)
        x = data[:, 0]
        f = data[:, 1]
        idx += 200
        cnt += 1
        name = pre + '-' + str(cnt) + '.png'
        if util.zero_data(x, f):
            continue
        if name in data_set:
            # print(name)
            for k in range(num):
                x_add, f_add = normalization_each_device(x, f, f_max, True)
                add_path = osp.join(base_path, pre + '-' + str(cnt) + '-' + str(k) + '.png')
                process(x_add, f_add, add_path)


# deprecated
def data_augmentation_from_origin(image_name_set, base_path):
    # order of magnitude
    label_num = len(image_name_set)
    num = int((4000 + random.randint(-200, 300)) / label_num)
    # get data source
    file_set = set()
    for image_name in image_name_set:
        file_set.add(image_name[0:4])
    # select data from all data
    filepath = 'data1'
    fileset = os.listdir(filepath)
    for fileName in fileset:
        # if this data file generate images
        if fileName[0:4] in file_set:
            if fileName.find('.xlsx') != -1:
                # file.enhance_excel(image_name_set)
                print('enhance_excel')
            elif fileName.find('.csv') != -1:
                augmentation_csv(osp.join(filepath, fileName), image_name_set, num, base_path)


def draw_line(img, x, f, color):
    for i in range(len(x)):
        if i == len(x) - 1:
            cv.line(img, (x[i], f[i]), (x[0], f[0]), color, 1, cv.LINE_AA)
        else:
            cv.line(img, (x[i], f[i]), (x[i + 1], f[i + 1]), color, 1, cv.LINE_AA)


def process_contrast(x_1, f_1, x_2, f_2, save_path):
    img = 255 * np.ones((256, 256, 3), np.uint8)
    draw_line(img, x_1, f_1, (255, 0, 0))
    draw_line(img, x_2, f_2, (0, 0, 255))
    img2 = cv.flip(img, 0, dst=None)
    cv.imwrite(save_path, img2)


def process(x_new, f_new, save_path):
    img = 255 * np.ones((256, 256, 3), np.uint8)
    draw_line(img, x_new, f_new, (255, 0, 0))
    img2 = cv.flip(img, 0, dst=None)
    cv.imwrite(save_path, img2)


def process_with_axis(x, f, save_path, max_f):
    # draw
    plt.rcParams['figure.figsize'] = (2.56, 2.56)
    plt.rcParams['savefig.dpi'] = 100
    plt.ylim(min(f) - 20, max(f) + 20)
    # plt.ylim(0, 256)
    plt.plot(x, f, '-', color='#000000')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def process_from_image(temp_list, save_path):
    arr = 255 * np.ones((256, 256, 3), np.uint8)
    for cnt in range(len(temp_list) - 1):
        cv.line(arr, temp_list[cnt], temp_list[cnt + 1], (255, 0, 0), 1, cv.LINE_AA)
    cv.line(arr, temp_list[-1], temp_list[0], (255, 0, 0), 1, cv.LINE_AA)
    cv.imwrite(save_path, arr)


def normalization_each_device(x, f, f_max, is_add):
    length = len(f)
    x_max = max(x)
    # f_max = max_value
    k_x = 253 / x_max
    k_f = 253 / f_max
    x_new = np.empty([length, 1], dtype=int)
    f_new = np.empty([length, 1], dtype=int)
    for i in range(length):
        add = random.randint(0, 1) if is_add and random.randint(0, 1) <= 0.5 else 0
        # add = random.randint(0, 1) if is_add else 0
        x_new[i] = int(x[i] * k_x) + 1
        f_new[i] = int((f[i] + add) * k_f) + 1
    return x_new, f_new


def normalization_each_image(x, f):
    length = len(f)
    x_max = max(x)
    f_max = max(f)
    k_x = 253 / x_max
    k_f = 253 / f_max
    x_new = np.empty([length, 1], dtype=int)
    f_new = np.empty([length, 1], dtype=int)
    move_step = (max(f) + min(f)) * 0.5 * k_f - 128
    for i in range(length):
        x_new[i] = int(x[i] * k_x) + 1
        f_new[i] = int(f[i] * k_f - move_step) + 1
    return x_new, f_new


def generate_map(image_name, x_new, f_new, content_list):
    image_dict = {"image_name": image_name,
                  "x": ','.join(str(x_new.tolist()).replace(']', '').replace('[', '').split(',')),
                  "f": ','.join(str(f_new.tolist()).replace(']', '').replace('[', '').split(','))}
    content_list.append(image_dict)


class Generate:

    def __init__(self, path=None, max_x=None, max_f=None):
        self.filepath = path
        # self.max_x = max_x
        # self.max_f = max_f

    def generate_excel(self, base_path):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        exception_not_match = 0
        exception_null = 0
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        image_cnt = 0
        for cnt in range(len(data)):
            # for cnt in range(2):
            # skip exception
            if util.nan_data(data[cnt, 0]) | util.nan_data(data[cnt, 1]):
                exception_null += 1
                print(pre + '-' + str(cnt) + ':null')
                continue
            x = data[cnt, 0].split(',')
            f = data[cnt, 1].split(',')
            if not util.match_data(x, f):
                exception_not_match += 1
                print(pre + '-' + str(cnt) + ':not match, x has ' + str(len(x)) + 'points and f has ' + str(
                    len(f)) + 'points')
                continue
            x_array = np.array(x, dtype=float)
            f_array = np.array(f, dtype=float)
            if util.zero_data(x_array, f_array):
                exception_zero += 1
                print(pre + '-' + str(cnt) + ':zero')
                continue
            # normalization_max
            x_new, f_new = normalization_each_image(x_array, f_array)
            # normalization_100
            # x_new, f_new = self.normalization_each_device(x_array, f_array, 1000, False)
            # draw and save
            save_path = osp.join(base_path, pre + '-' + str(cnt) + '.png')
            process(x_new, f_new, save_path)
            image_cnt += 1
        print(pre + ' num = ' + str(exception_not_match) + ' ; ' + str(exception_null) + ' ; ' + str(
            exception_zero) + ' ; ' + str(image_cnt))

    def generate_csv(self, base_path):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        print(pre + ' start, max_f = ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        # image_cnt = 0
        while maxlength - 200 * cnt > 0:
            # while cnt < 5:
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            # skip exception
            if util.zero_data(x, f):
                exception_zero += 1
                # print(pre + '-' + str(cnt) + ':zero')
                continue
            # normalization_max
            # x_new, f_new = self.normalization_each_image(x, f)
            # normalization_100
            x_new, f_new = normalization_each_device(x, f, f_max, False)
            # draw and save
            save_path = osp.join(base_path, pre + '-' + str(cnt) + '.png')
            process(x_new, f_new, save_path)
            # image_cnt += 1
        # print(pre + ' num = ' + str(exception_zero) + ' ; ' + str(image_cnt))

    def generate_csv_i(self, base_path):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv_i()
        maxlength = len(file)
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        print(pre + ' start, max_f = ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        # image_cnt = 0
        while maxlength - 200 * cnt > 0:
            # while cnt < 5:
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            # skip exception
            if util.zero_data(x, f):
                exception_zero += 1
                # print(pre + '-' + str(cnt) + ':zero')
                continue
            # normalization_max
            # x_new, f_new = self.normalization_each_image(x, f)
            # normalization_100
            x_new, f_new = normalization_each_device(x, f, f_max, False)
            # draw and save
            save_path = osp.join(base_path, pre + '-' + str(cnt) + '.png')
            x_ii = range(0, 200)
            process_with_axis(x_ii, f_new, save_path, None)
            # process(x_new, f_new, save_path)
            # image_cnt += 1
        # print(pre + ' num = ' + str(exception_zero) + ' ; ' + str(image_cnt))

    def generate_image(self, base_path):
        loadfile = load.LoadData(self.filepath)
        image_data = loadfile.get_data_from_image()
        pre = osp.splitext(osp.basename(self.filepath))[0]
        temp_data = image_data
        temp_list = []
        for data in temp_data:
            temp_list.append(data)
        save_path = osp.join(base_path, pre + '.png')
        # process_from_image(temp_list, save_path)
        x = []
        f = []
        for i in temp_list:
            x.append(i[0])
            f.append(i[1])
        process_with_axis(x, f, save_path, None)

    def generate_csv_with_axis(self, base_path):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        print(pre + ' start, max_f = ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        # image_cnt = 0
        while maxlength - 200 * cnt > 0:
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            if cnt == 3:
                if util.zero_data(x, f):
                    exception_zero += 1
                    continue
                # x_new = np.empty([200, 1], dtype=int)
                # f_new = np.empty([200, 1], dtype=int)
                # for i in range(200):
                #     x_new[i] = int(x[i])
                #     f_new[i] = int(f[i])
                save_path = osp.join(base_path, pre + '-' + str(cnt) + '.png')
                process_with_axis(x, f, save_path, f_max)

    def generate_excel_map(self, content_list):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        pre = osp.splitext(osp.basename(self.filepath))[0]
        for cnt in range(len(data)):
            # for cnt in range(2):
            # skip exception
            if util.nan_data(data[cnt, 0]) | util.nan_data(data[cnt, 1]):
                continue
            x = data[cnt, 0].split(',')
            f = data[cnt, 1].split(',')
            if not util.match_data(x, f):
                continue
            x_array = np.array(x, dtype=float)
            f_array = np.array(f, dtype=float)
            if util.zero_data(x_array, f_array):
                continue
            # normalization_max
            x_new, f_new = normalization_each_image(x_array, f_array)
            image_name = pre + '-' + str(cnt) + '.png'
            generate_map(image_name, x_new, f_new, content_list)

    def generate_csv_map(self, content_list):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        print(pre + ' start, max_f = ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        while maxlength - 200 * cnt > 0:
            # while cnt < 5:
            image_name = pre + '-' + str(cnt) + '.png'
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            # skip exception
            if util.zero_data(x, f):
                exception_zero += 1
                print(pre + '-' + str(cnt) + ':zero')
                continue
            # normalization_max
            # x_new, f_new = self.normalization_each_image(x, f)
            # normalization_100
            x_new, f_new = normalization_each_device(x, f, f_max, False)
            generate_map(image_name, x_new, f_new, content_list)

    def generate_image_map(self, content_list):
        image_path_set = os.listdir(self.filepath)
        for image in image_path_set:
            loadfile = load.LoadData(osp.join(self.filepath, image))
            image_data = loadfile.get_data_from_image()
            pre = osp.splitext(osp.basename(osp.join(self.filepath, image)))[0]
            length = len(image_data)
            x_new = np.empty([length, 1], dtype=int)
            f_new = np.empty([length, 1], dtype=int)
            for i in range(length):
                x_new[i] = image_data[i][0]
                f_new[i] = image_data[i][1]
            image_name = pre + '.png'
            # image_name = pre + '.jpg'
            generate_map(image_name, x_new, f_new, content_list)
            print(image_name)

    # deprecated
    def get_standard_excel(self, standard):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        pre = osp.splitext(osp.basename(self.filepath))[0]
        for cnt in range(len(data)):
            # skip exception
            if util.nan_data(data[cnt, 0]) | util.nan_data(data[cnt, 1]):
                continue
            x = data[cnt, 0].split(',')
            f = data[cnt, 1].split(',')
            if not util.match_data(x, f):
                continue
            x_array = np.array(x, dtype=float)
            f_array = np.array(f, dtype=float)
            if util.zero_data(x_array, f_array):
                continue
            name = pre + '-' + str(cnt) + '.png'
            if name == standard:
                x_standard, f_standard = normalization_each_image(x_array, f_array)
                return x_standard, f_standard

    # deprecated
    def get_standard_csv(self, standard):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        # print('standard image source is' + pre + ', max_f is ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        while maxlength - 200 * cnt > 0:
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            name = pre + '-' + str(cnt) + '.png'
            if util.zero_data(x, f):
                continue
            if name == standard:
                x_standard, f_standard = normalization_each_device(x, f, f_max, False)
                return x_standard, f_standard

    # deprecated
    def generate_excel_contrast(self, data_set, x_standard, f_standard):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        pre = osp.splitext(osp.basename(self.filepath))[0]
        for cnt in range(len(data)):
            # skip exception
            if util.nan_data(data[cnt, 0]) | util.nan_data(data[cnt, 1]):
                continue
            x = data[cnt, 0].split(',')
            f = data[cnt, 1].split(',')
            if not util.match_data(x, f):
                continue
            x_array = np.array(x, dtype=float)
            f_array = np.array(f, dtype=float)
            if util.zero_data(x_array, f_array):
                continue
            name = pre + '-' + str(cnt) + '.png'
            if name in data_set:
                x_new, f_new = normalization_each_image(x_array, f_array)
                # standard_name = standard.replace('.png')
                save_path = 'image2\\plus\\' + pre + '-' + str(cnt) + 'vs.png'
                process_contrast(x_new, f_new, x_standard, f_standard, save_path)

    # deprecated
    def generate_csv_contrast(self, data_set, x_standard, f_standard):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        pre = osp.splitext(osp.basename(self.filepath))[0]
        f_max = max(np.array(file)[:, 1])
        print(pre + ' start, max_f = ' + str(f_max))
        # split
        cnt = 0
        idx = 0
        while maxlength - 200 * cnt > 0:
            temp = file[idx:idx + 200]
            data = np.array(temp)
            x = data[:, 0]
            f = data[:, 1]
            idx += 200
            cnt += 1
            name = pre + '-' + str(cnt) + '.png'
            if util.zero_data(x, f):
                continue
            if name in data_set:
                x_add, f_add = normalization_each_device(x, f, f_max, False)
                # standard_name = standard.replace('.png')
                save_path = 'image2\\plus\\' + pre + '-' + str(cnt) + 'vs.png'
                process_contrast(x_add, f_add, x_standard, f_standard, save_path)


def generate_i():
    file = Generate('D:\\pythonProject\\data1\\FD-1.csv')
    file.generate_csv_i('D:\\pythonProject\\data1\\')


if __name__ == '__main__':
    # generate_origin('D:\\pythonProject\\testt\\',
    #                 'D:\\pythonProject\\testt\\')
    generate_origin_map('D:\pythonProject\image\data1214',
                        'D:\\pythonProject\\image_data_map_image.csv')
    # generate_enhance('D:\\pythonProject\\image\\images')
    # data_augmentation_from_image('D:\\graduationproject\DataPreparation\\1119test\\data', 4500,
    #                              'D:\\pythonProject\\image\\origin_images\\generate')
    # parse_excel('D:\\pythonProject\\data\\test\\大港油田示功图测试数据.xls', 'D:\\pythonProject\\txtdata')
    # generate_i()
