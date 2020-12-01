"""
created on 2020/10/26 12:10
@author:yuka
@note:generate origin indicator diagram
      1.get data from a excel or csv file
      2.normalization
      3.draw indicator diagram
      4.save image as a png file
      5.generate origin map:png file name; x pixel; f pixel
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


def generate_origin(filepath, base_path):
    fileset = os.listdir(filepath)
    for fileName in fileset:
        file = Generate(osp.join(filepath, fileName))
        if fileName.find('.xlsx') != -1:
            file.generate_excel(base_path)
        elif fileName.find('.csv') != -1:
            file.generate_csv(base_path)
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
    pf = pd.DataFrame(content_list, columns=["image_name", "x", "f"])
    pf.to_csv(save_path)


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
    # plt.ylim(min(f) - 20, max(f) + 10)
    plt.ylim(0, max_f)
    plt.plot(x, f, '-', color='#000000')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


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
            x_new, f_new = self.normalization_each_image(x_array, f_array)
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
            x_new, f_new = self.normalization_each_device(x, f, f_max, False)
            # draw and save
            save_path = osp.join(base_path, pre + '-' + str(cnt) + '.png')
            process(x_new, f_new, save_path)
            # image_cnt += 1
        # print(pre + ' num = ' + str(exception_zero) + ' ; ' + str(image_cnt))

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
            x_new, f_new = self.normalization_each_image(x_array, f_array)
            image_dict = {"image_name": pre + '-' + str(cnt) + '.png',
                          "x": ','.join(str(x_new.tolist()).replace(']', '').replace('[', '').split(',')),
                          "f": ','.join(str(f_new.tolist()).replace(']', '').replace('[', '').split(','))}
            content_list.append(image_dict)

    def generate_csv_map(self, content_list):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        exception_zero = 0
        pre = osp.splitext(osp.basename(self.filepath))[0]
        # filenames = self.filepath.split('\\')
        # pre = filenames[len(filenames) - 1].replace('.csv', '')
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
            x_new, f_new = self.normalization_each_device(x, f, f_max, False)
            image_dict = {"image_name": pre + '-' + str(cnt) + '.png',
                          "x": ','.join(str(x_new.tolist()).replace(']', '').replace('[', '').split(',')),
                          "f": ','.join(str(f_new.tolist()).replace(']', '').replace('[', '').split(','))}
            content_list.append(image_dict)

    @staticmethod
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

    @staticmethod
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

    # deprecated
    def get_standard_excel(self, standard):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        filenames = self.filepath.split('\\')
        pre = filenames[len(filenames) - 1].replace('.xlsx', '')
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
                x_standard, f_standard = self.normalization_each_image(x_array, f_array)
                return x_standard, f_standard

    # deprecated
    def get_standard_csv(self, standard):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        filenames = self.filepath.split('\\')
        pre = filenames[len(filenames) - 1].replace('.csv', '')
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
                x_standard, f_standard = self.normalization_each_device(x, f, f_max, False)
                return x_standard, f_standard

    # deprecated
    def generate_excel_contrast(self, data_set, x_standard, f_standard):
        loadfile = load.LoadData(self.filepath)
        data = loadfile.read_excel()
        filenames = self.filepath.split('\\')
        pre = filenames[len(filenames) - 1].replace('.xlsx', '')
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
                x_new, f_new = self.normalization_each_image(x_array, f_array)
                # standard_name = standard.replace('.png')
                save_path = 'image2\\plus\\' + pre + '-' + str(cnt) + 'vs.png'
                process_contrast(x_new, f_new, x_standard, f_standard, save_path)

    # deprecated
    def generate_csv_contrast(self, data_set, x_standard, f_standard):
        loadfile = load.LoadData(self.filepath)
        file = loadfile.read_csv()
        maxlength = len(file)
        filenames = self.filepath.split('\\')
        pre = filenames[len(filenames) - 1].replace('.csv', '')
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
                x_add, f_add = self.normalization_each_device(x, f, f_max, False)
                # standard_name = standard.replace('.png')
                save_path = 'image2\\plus\\' + pre + '-' + str(cnt) + 'vs.png'
                process_contrast(x_add, f_add, x_standard, f_standard, save_path)


if __name__ == '__main__':
    generate_origin('D:\\pythonProject\\data1', 'D:\\pythonProject\\5')
    generate_origin_map('D:\\pythonProject\\data1', 'D:\\pythonProject\\image_data_map.csv')
