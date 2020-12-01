"""
created on 2020/12/1 14:36
@author:yuka
@note:generate test sample
      1.generate 2-classify train/test sample
      2.generate triplet test sample
        2.1 generate origin sample from 2-classify test sample
        2.2 delete_duplicate
        2.3 print into a txt file:a.png b.png label
      3.select val sample
      4.data augmentation
        4.1 from image
        4.2 from pixel map
        4.3 from origin(deprecated)
"""

import generate_origin
import load
import util
import os
import os.path as osp
import random
import numpy as np
import cv2 as cv
from shutil import move


def generate_contrast_plus_train(img_path, base_path):
    image_path_set = os.listdir(img_path)
    print(img_path + ' has ' + str(len(image_path_set)) + ' labels ')
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    for label in image_path_set:
        image_name_set = os.listdir(osp.join(img_path, label))
        label_num = len(image_name_set)
        # choose first image as standard image todo standard image is not static
        standard_image = image_name_set[0]
        print(label + ' has ' + str(label_num) + ' images, standard image is ' + standard_image)
        index_standard = dict_index.get(standard_image)
        temp_image_set = list(image_name_set)
        temp_image_set.remove(standard_image)
        for image_name in temp_image_set:
            index_image = dict_index.get(image_name)
            print('index_1: ' + str(index_standard) + ', index_2: ' + str(index_image))
            if index_standard is None or index_image is None:
                continue
            print(label + ': generate ' + image_name + ' with ' + standard_image)
            x_1, f_1 = load.get_pixel_by_index(index_standard, pf)
            x_2, f_2 = load.get_pixel_by_index(index_image, pf)
            pre = image_name.replace('.png', '')
            save_path = osp.join(base_path, pre + 'tp.png')
            generate_origin.process_contrast(x_1, f_1, x_2, f_2, save_path)


def generate_contrast_minus_train(img_path, base_path):
    image_path_set = os.listdir(img_path)
    label_cnt = len(image_path_set)
    print(img_path + ' has ' + str(label_cnt) + ' labels ')
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    for label in image_path_set:
        temp_label_set = list(image_path_set)
        temp_label_set.remove(label)
        image_name_set = os.listdir(osp.join(img_path, label))
        for image_name in image_name_set:
            minus_label = temp_label_set[random.randint(0, len(temp_label_set) - 1)]
            minus_name_set = os.listdir(img_path + '\\' + minus_label)
            minus_name = minus_name_set[random.randint(0, len(minus_name_set) - 1)]
            index_1 = dict_index.get(image_name)
            index_2 = dict_index.get(minus_name)
            print('index_1: ' + str(index_1) + ', index_2: ' + str(index_2))
            if index_1 is None or index_2 is None:
                continue
            print(label + ': generate ' + image_name + ' with ' + minus_name)
            x_1, f_1 = load.get_pixel_by_index(index_1, pf)
            x_2, f_2 = load.get_pixel_by_index(index_2, pf)
            pre = image_name.replace('.png', '')
            save_path = osp.join(base_path, pre + 'tm.png')
            generate_origin.process_contrast(x_1, f_1, x_2, f_2, save_path)


# 250~590 todo
def generate_contrast_plus_test(path, base_path):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    image_path_set = os.listdir(path)
    for label in image_path_set:
        image_name_set = os.listdir(osp.join(path, label))
        num = random.randint(400, 700)
        image_name1 = image_name_set[0]
        temp_label_set = list(image_name_set)
        temp_label_set.remove(image_name1)
        for i in range(num):
            image_name2 = temp_label_set[random.randint(0, len(temp_label_set) - 1)]
            index_1 = dict_index.get(image_name1)
            index_2 = dict_index.get(image_name2)
            print('index_1: ' + str(index_1) + ', index_2: ' + str(index_2))
            if index_1 is None or index_2 is None:
                continue
            x_1, f_1 = load.get_pixel_by_index(index_1, pf)
            x_2, f_2 = load.get_pixel_by_index(index_2, pf)
            pre1 = image_name1.replace('.png', '')
            pre2 = image_name2.replace('.png', '')
            save_path = osp.join(base_path, pre1 + '~' + pre2 + '~' + str(i) + 'tp.png')
            generate_origin.process_contrast(x_1, f_1, x_2, f_2, save_path)


# todo
def generate_contrast_minus_test(image_name, path, base_path):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    image_path_set = os.listdir(path)
    num = random.randint(800, 1000)
    for i in range(num):
        minus_label = image_path_set[random.randint(0, len(image_path_set) - 1)]
        minus_name_set = os.listdir(osp.join(path, minus_label))
        minus_name = minus_name_set[random.randint(0, len(minus_name_set) - 1)]
        index_1 = dict_index.get(image_name)
        index_2 = dict_index.get(minus_name)
        print('index_1: ' + str(index_1) + ', index_2: ' + str(index_2))
        if index_1 is None or index_2 is None:
            continue
        x_1, f_1 = load.get_pixel_by_index(index_1, pf)
        x_2, f_2 = load.get_pixel_by_index(index_2, pf)
        pre1 = image_name.replace('.png', '')
        pre2 = minus_name.replace('.png', '')
        save_path = osp.join(base_path, pre1 + '~' + pre2 + '~' + str(i) + 'tm.png')
        generate_origin.process_contrast(x_1, f_1, x_2, f_2, save_path)


def generate_single_contrast_by_name(image_name1, image_name2, base_path):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    # file = open('D:\\graduationproject\\ver3\\similarity\\cbir\\error.txt')
    # for line in file:
    #     content = line.split(' ')
    #     image_name1 = content[0]
    #     image_name2 = content[1]
    #     label = content[2].replace('\n', '')
    index_1 = dict_index.get(image_name1)
    index_2 = dict_index.get(image_name2)
    print(str(index_1) + ';' + str(index_2))
    # if index_1 is None or index_2 is None:
    #     continue
    x_1, f_1 = load.get_pixel_by_index(index_1, pf)
    x_2, f_2 = load.get_pixel_by_index(index_2, pf)
    save_path = osp.join(base_path, image_name1 + '~' + image_name2 + '.png')
    generate_origin.process_contrast(x_2, f_2, x_1, f_1, save_path)
    # file.close()


def generate_triplet_origin_image(save_path, source_path):
    loadfile = load.LoadData('D:\\pythonProject\\image_data_map.csv')
    pf, dict_index = loadfile.get_pixel_data()
    image_path_set = os.listdir(source_path)
    image_set = set()
    for image in image_path_set:
        temp = image.split('~')
        image_set.add(temp[0] + '.png')
        image_set.add(temp[1] + '.png')
    print('origin image num: ' + str(len(image_set)))
    for image_name in image_set:
        index = dict_index.get(image_name)
        if index is None:
            continue
        x, f = load.get_pixel_by_index(index, pf)
        generate_origin.process(x, f, osp.join(save_path, image_name))
        print('origin image: ' + image_name)


def delete_duplicate(path):
    cnt = 0
    all_sample = set()
    image_set = os.listdir(path)
    for image in image_set:
        temp = image.split('~')
        temp1 = temp[0] + '~' + temp[1]
        temp2 = temp[1] + '~' + temp[0]
        if temp1 in all_sample or temp2 in all_sample:
            print(path + image)
            cnt += 1
            os.remove(path + '\\' + image)
        all_sample.add(temp1)
        all_sample.add(temp1)
    print(str(cnt))


def generate_similarity_list(folder_path, save_file):
    delete_duplicate(folder_path)

    cnt = 0
    with open(save_file, 'a') as fw:
        # fw.write(data_path + '\n')
        image_path_set = os.listdir(folder_path)
        for image in image_path_set:
            temp = image.split('~')
            image1 = temp[0]
            image2 = temp[1]
            label = 0 if image.find('tm') != -1 else 1
            each_line = image1 + ' ' + image2 + ' ' + str(label) + '\n'
            fw.write(each_line)
            cnt += 1
    print(cnt)


def select_val_sample(train_path, val_path, base_num):
    image_path_set = os.listdir(train_path)
    print(train_path + ' has ' + str(len(image_path_set)) + ' labels ')
    for label in image_path_set:
        image_name_set = os.listdir(osp.join(train_path, label))
        # print('first: '+str(len(image_name_set)))
        num = int(base_num + random.randint(-40, 30))
        while num > 0:
            # print(str(len(image_name_set)))
            index = random.randint(0, len(image_name_set))
            image_name = image_name_set[index]
            image_name_set.remove(image_name)
            # shutil.copyfile(old_name, new_name)
            src_path = osp.join(train_path, label, image_name)
            dst_path = osp.join(val_path, label)
            move(src_path, dst_path)
            num -= 1


def process_from_image(temp_list, save_path):
    arr = 255 * np.ones((256, 256, 3), np.uint8)
    for cnt in range(len(temp_list) - 1):
        cv.line(arr, temp_list[cnt], temp_list[cnt + 1], (255, 0, 0), 1, cv.LINE_AA)
    cv.line(arr, temp_list[-1], temp_list[0], (255, 0, 0), 1, cv.LINE_AA)
    cv.imwrite(save_path, arr)


def data_augmentation_from_image(img_path, base_num, base_path):
    image_set = os.listdir(img_path)
    for image in image_set:
        loadfile = load.LoadData(img_path)
        image_data = loadfile.get_data_from_image()
        # image_data = generate.get_data_from_image(path + '\\' + image)
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
            generate_origin.process(x, f_add, add_path)


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
                x_add, f_add = generate_origin.normalization_each_device(x, f, f_max, True)
                add_path = osp.join(base_path, pre + '-' + str(cnt) + '-' + str(k) + '.png')
                generate_origin.process(x_add, f_add, add_path)


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


if __name__ == '__main__':
    generate_contrast_plus_train('D:\\pythonProject\\image1\\test', 'D:\\pythonProject\\image2')
    # generate_contrast_minus_test('FD-1-360.png', 'D:\\pythonProject\\image1\\testset', 'D:\\pythonProject\\image2')
    # generate_single_contrast_by_name('FD-1-360.png', 'FD-1-380.png',
    #                                  'D:\\graduationproject\\ver3\\similarity\\cbir\\error')
    # generate_triplet_origin_image('D:\\pythonProject\\image2\\SimilarityDetection\\origin',
    #                               'D:\\pythonProject\\image2\\SimilarityDetection\\all')
    # generate_similarity_list('D:\\pythonProject\\image2\\SimilarityDetection\\origin',
    #                          'D:\\pythonProject\\image2\\SimilarityDetection\\data.txt')
    # select_val_sample('image\\oilsimilarity\\generate_from_image\\train',
    #                   'image\\oilsimilarity\\generate_from_image\\val\\', 150)
    # generate_enhance('D:\\pythonProject\\image\\images')
    # data_augmentation_from_image('D:\\graduationproject\DataPreparation\\1119test\\data', 4500,
    #                              'D:\\pythonProject\\image\\origin_images\\generate')
