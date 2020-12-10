"""
created on 2020/12/1 9:48
@author:yuka Shar
@note:result analysis:
      1.generate result from train log
        1.1 transfer train log to csv file: log_name,train_iter,train_acc,train_loss,val_iter,val_acc,val_loss
            1.1.1 grep test
            1.1.2 grep train
        1.2 draw result
            1.2.1 single log - single param
            1.2.2 multiple log -single param
            1.2.3
      2.2-classify test error set analysis
      3.2-classify test set modify:
        3.1 delete error sample from total sample
        3.2 change sample label(attached to file name)
        3.3 copy modified sample to total sample
      4.generate result from tes log(single or multiple)
      5.find common error set between 2-classify test and triplet test
"""

import re
import os
import os.path as osp
import shutil
from shutil import move
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random

ch = {
    'iter': '迭代次数',
    'acc': '准确度',
    'loss': '损失率',
    'train_acc': '训练集准确度',
    'train_loss': '训练集损失率',
    'val_acc': '验证集准确度',
    'val_loss': '验证集损失率',
    'test_loss': '测试集损失率'
}


def generate_log_result(path_to_log, flag):
    iteration = []
    acc = []
    loss = []
    regex_iteration = re.compile('Iteration (\d+)')
    is_iteration = -1

    with open(path_to_log) as f:
        for line in f:
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                is_iteration = float(iteration_match.group(1))
            if is_iteration == -1:
                # 日志尚未扫描到迭代部分
                continue

            if flag == 0:
                # generate train result
                if 'Iteration' in line and ', loss = ' in line:
                    print(line)
                    iteration.append(int(re.findall(r"Iteration (.+?)[ (|,]", line)[0]))
                    loss.append(float(re.findall(r", loss = (.+?)$", line)[0]))
                if 'Train net output #0' in line:
                    print(line)
                    acc.append(float(re.findall(r"accuracy = (.+?)$", line)[0]))
            else:
                # generate val result
                if 'Testing net (#0)' in line:
                    print(line)
                    iteration.append(int(re.findall(r"Iteration (.+?),", line)[0]))
                if 'Test net output #0' in line:
                    print(line)
                    acc.append(float(re.findall(r"accuracy = (.+?)$", line)[0]))
                if 'Test net output #1' in line:
                    print(line)
                    loss.append(float(re.findall(r"loss = (.+?) [(]", line)[0]))

    return iteration, acc, loss


def train_log2csv(log_path, base_path, log_basename):
    train_iter, train_acc, train_loss = generate_log_result(log_path, 0)
    val_iter, val_acc, val_loss = generate_log_result(log_path, 1)
    with open(base_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        if len(open(base_path, 'r').readlines()) == 0:
            # writerow when create
            writer.writerow(('log_name',
                             'train_iter', 'train_acc', 'train_loss',
                             'val_iter', 'val_acc', 'val_loss'))
        tmp = [log_basename, train_iter, train_acc, train_loss, val_iter, val_acc, val_loss]
        writer.writerow(tmp)


def random_marker(length):
    result = []
    marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4',
                   's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
    marker_index = random.sample(range(0, 22), length)
    for index in range(length):
        result.append(marker_list[marker_index[index]])
    return result


def plot_training_log(x_label, y_label, series, save_path):
    length = len(series)
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['savefig.dpi'] = 100
    marker_list = random_marker(length)
    legends = []

    for index, d in enumerate(series):
        legends.append(d['legend'])
        color = [random.random(), random.random(), random.random()]
        marker = marker_list[index]
        x_value = list(map(eval, d['x'][1:len(d['x']) - 1].split(',')))
        y_value = list(map(eval, d['y'][1:len(d['y']) - 1].split(',')))
        # acc last loss
        if len(x_value) != len(y_value):
            x_value = x_value[0:len(x_value)-1]
        plt.plot(x_value, y_value, marker=marker, color=color, linewidth=0.75)

    plt.legend(legends)
    plt.title(ch[y_label] + '与' + ch[x_label] + "的关系")
    plt.xlabel(ch[x_label])
    plt.ylabel(ch[y_label])
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# todo 判空
def draw_single_param(x_label, y_label, names, save_path):
    origin = pd.read_csv('log_result.csv')
    # 设置下标
    origin.set_index("log_name", inplace=True)
    # 获取指定行
    data = origin.loc[names]
    x_list = data[x_label]
    y_list = data[y_label]
    series = []

    for index, n in enumerate(names):
        s = dict()
        s['legend'] = n + ch[y_label]
        s['x'] = x_list.iloc[index]
        s['y'] = y_list.iloc[index]
        series.append(s)

    plot_training_log(x_label.split('_')[1], y_label.split('_')[1], series, save_path)


# todo 判空
def draw_multiple_param(data, name, save_path):
    origin = pd.read_csv('log_result.csv')
    # 设置下标
    origin.set_index("log_name", inplace=True)
    # 获取指定行
    record = origin.loc[name]
    series = []

    for d in data:
        s = dict()
        s['legend'] = name + ch[d[1]]
        s['x'] = record[d[0]]
        s['y'] = record[d[1]]
        series.append(s)

    plot_training_log(data[0][0].split('_')[1], data[0][1].split('_')[1], series, save_path)


def draw_test_result(paths, save_path):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['savefig.dpi'] = 100
    length = len(paths)
    marker_list = random_marker(length)
    legends = []

    for index, path in enumerate(paths):
        legends.append(path.split('_')[0] + '测试集损失率')
        data_list = open(path, 'r').readlines()
        x = []
        y = []
        for data in data_list:
            data.strip()
            x.append(int(data.split(' ')[0]))
            y.append(float(data.split(' ')[1]) / 8833)
        color = [random.random(), random.random(), random.random()]
        marker = marker_list[index]
        plt.plot(x, y, marker=marker, color=color, linewidth=0.75)

    plt.legend(legends)
    plt.title("损失率与迭代次数的关系")
    plt.xlabel('迭代次数')
    plt.ylabel('损失率')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# deprecated
def generate_test_result(path, save_path):
    df = pd.read_excel(path)
    iter_time = df['Model'].tolist()
    accurency = df['accurency'].tolist()
    loss = df['loss'].tolist()
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['savefig.dpi'] = 100
    # plt.ylim(min(f) - 20, max(f) + 10)
    plt.plot(iter_time, loss, linestyle='-', marker='^', color='#ADD8E6')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# deprecated
def process_contrast_result(x1, f1, x2, f2, save_path):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['savefig.dpi'] = 100
    # plt.ylim(min(f) - 20, max(f) + 10)

    plt.plot(x1, f1, marker='o', color='red', linewidth=2.0, linestyle='--')
    plt.plot(x2, f2, marker='>', color='blue', linewidth=2.0, linestyle='-')
    plt.legend(['训练集', '验证集'])
    plt.title("损失率与迭代次数的关系")
    plt.xlabel('迭代次数')
    plt.ylabel('损失率')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# deprecated
def generate_contrast_result(txt_path, save_file):
    file = open(txt_path)
    val_loss_list = []
    train_loss_list = []
    for line in file:
        if line.find('Test net output #1: loss =') != -1:
            temp = line.split('(')[1]
            val_loss = temp[5:len(temp) - 6]
            print('val loss is ' + val_loss)
            val_loss_list.append(float(val_loss))
        elif line.find('Train net output #1: loss =') != -1:
            temp = line.split('(')[1]
            train_loss = temp[5:len(temp) - 6]
            print('train loss is ' + train_loss)
            train_loss_list.append(float(train_loss))
    file.close()
    val_iter = list(range(0, 100000, 4000))
    train_iter = list(range(0, 100000, 2000))
    process_contrast_result(train_iter, train_loss_list, val_iter, val_loss_list, save_file)


# deprecated
def process_single_result(x1, f1, save_path):
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (6.4, 4.8)
    plt.rcParams['savefig.dpi'] = 100
    # plt.ylim(min(f) - 20, max(f) + 10)

    plt.plot(x1, f1, marker='o', color='red', linewidth=2.0, linestyle='-')
    plt.legend(['训练集'])
    plt.title("损失率与迭代次数的关系")
    plt.xlabel('迭代次数')
    plt.ylabel('损失率')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# deprecated
def generate_single_result(txt_path, save_file, min_iter, max_iter, interval):
    file = open(txt_path)
    loss_list = []
    for line in file:
        if line.find('Iteration') != -1 and line.find('loss') != -1:
            temp_len = len(line.split(','))
            temp = line.split(',')[temp_len - 1]
            loss = temp[7:len(temp) - 1]
            print('loss is ' + loss)
            loss_list.append(float(loss))
    file.close()
    iteration = list(range(min_iter, max_iter + 1, interval))
    process_single_result(iteration, loss_list, save_file)


def get_minus_and_plus_num(image_name_set, label, error_set, path):
    minus_num = 0
    plus_num = 0
    for image_name in image_name_set:
        if image_name.find('tm') != -1:
            minus_num += 1
            if label == '1':
                error_set.add(image_name)
                shutil.copyfile(osp.join(path, label, image_name), osp.join(path, 'error', image_name))
        elif image_name.find('tp') != -1 or image_name.find('vs') != -1:
            plus_num += 1
            if label == '0':
                error_set.add(image_name)
                shutil.copyfile(osp.join(path, label, image_name), osp.join(path, 'error', image_name))
    print(label + ' :minus: ' + str(minus_num) + ', plus: ' + str(plus_num))
    return minus_num, plus_num


# already added to 2_classify_img_with_error.py
def select_error(path, save_file):
    label_set = os.listdir(path)
    error_set = set()
    for label in label_set:
        image_set = os.listdir(osp.join(path, label))
        get_minus_and_plus_num(image_set, label, error_set, path)
    print(str(len(error_set)))

    with open(save_file, 'a') as fw:
        fw.write(save_file + '\n')
        for image in error_set:
            each_line = image + '\n'
            fw.write(each_line)


def delete_error(error_path, path):
    error_set = os.listdir(error_path)
    image_set = os.listdir(path)
    for image in image_set:
        if image in error_set:
            print(image)
            os.remove(osp.join(path, image))


def change_sample(error_path):
    error_set = os.listdir(error_path)
    for image in error_set:
        if image.find('tp') != -1:
            new_name = image.replace('tp', 'tm')
            os.rename(error_path + image, error_path + new_name)
        if image.find('tm') != -1:
            new_name = image.replace('tm', 'tp')
            os.rename(error_path + image, error_path + new_name)


def modify_sample(error_path, path):
    delete_error(error_path, path)
    change_sample(error_path)
    error_set = os.listdir(error_path)
    for image in error_set:
        src_path = osp.join(error_path, image)
        dst_path = path
        move(src_path, dst_path)


def find_common_result(error_path_2, error_path_3):
    # dict_data = dict()
    # file = open('D:\\graduationproject\\ver3\\similarity\\cbir\\error.txt')
    # for line in file:
    #     content = line.split(' ')
    #     image_name1 = content[0]
    #     image_name2 = content[1]
    #     label = content[2].replace('\n', '')
    #     key = image_name1 + '~' + image_name2
    #     value = label.replace('.png', '')
    #     dict_data[key] = value
    # file.close()

    image_all_set = os.listdir(error_path_2)
    dict_error = dict()
    for image in image_all_set:
        content = image.split('~')
        image_name1 = content[0]
        image_name2 = content[1]
        label = content[2].replace('\n', '')
        key = image_name1 + '~' + image_name2
        value = label.replace('.png', '')
        dict_error[key] = value

    cnt = 0
    image_set = os.listdir(error_path_3)
    for image in image_set:
        content = image.split('~')
        image_name1 = content[0]
        image_name2 = content[1]
        label = content[2].replace('\n', '')
        key = image_name1 + '~' + image_name2
        error_label = dict_error.get(key, None)
        if error_label is not None:
            cnt += 1
            print(key)
    print(cnt)


if __name__ == "__main__":
    # generate_contrast_result('D:\\graduationproject\\ver3\\log2-18.txt',
    #                          'D:\\graduationproject\\ver3\\result2-18.png')
    # generate_single_result('D:\\graduationproject\\ver3\\log3-1.txt', 'D:\\graduationproject\\ver3\\result3-1.png',
    #                        0, 210000, 2000)
    # modify_sample('D:\\graduationproject\\ver3\compare\\1110test\\todo\\',
    #  'D:\\pythonProject\\image2\\SimilarityDetection\\all')
    # draw_test_result('D:\\graduationproject\\ver3\\compare\\test_analyse.xlsx',
    #                  'D:\\graduationproject\\ver3\\compare\\test_loss1.png')
    # find_common_result('D:\\graduationproject\\ver3\compare\\1110test\\error',
    #                    'D:\\graduationproject\\ver3\\similarity\\cbir\\error')
    # train_log2csv('D:\\graduationproject\\ver3\\similarity\\1117\\train3.log',
    #               'log_result.csv', 'triplet50_34')
    # draw_single_param('train_iter', 'train_loss', ['resnet50_34', 'triplet50_34'], 'test.png')
    draw_test_result(['resnet18_45_result.txt', 'shufflenetv2_45_result.txt'], 'test1.png')
    # draw_multiple_param([['train_iter', 'train_acc'], ['val_iter', 'val_acc']], 'resnet50_34', 'test2.png')
