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
"""

import generate_origin
import load
import os
import os.path as osp
import random
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
