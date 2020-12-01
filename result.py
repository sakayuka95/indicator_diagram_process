"""
created on 2020/12/1 9:48
@author:yuka
@note:result analysis:
      1.generate result from train log
      2.2-classify test error set analysis
      3.2-classify test set modify:
        3.1 delete error sample from total sample
        3.2 change sample label(attached to file name)
        3.3 copy modified sample to total sample
      4.draw 2-classify test result among all model snapshots
"""

import generate_origin
import os
import os.path as osp
import shutil
from shutil import move
import pandas as pd
import matplotlib.pyplot as plt


def generate_result_from_log(txt_path, save_file):
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
    generate_origin.process_result(train_iter, train_loss_list, val_iter, val_loss_list, save_file)


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


def draw_test_result(path, save_path):
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



if __name__ == "__main__":
    generate_result_from_log('D:\\graduationproject\\ver3\\log3.txt', 'D:\\graduationproject\\ver3\\result3.png')
    # modify_sample('D:\\graduationproject\\ver3\compare\\1110test\\todo\\',
    #  'D:\\pythonProject\\image2\\SimilarityDetection\\all')
    # draw_test_result('D:\\graduationproject\\ver3\\compare\\test_analyse.xlsx', 'D:\\graduationproject\\ver3\\compare\\test_loss1.png')
