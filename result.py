"""
created on 2020/12/1 9:48
@author:yuka
@note:generate result from train log
"""

import generate


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
    generate.process_result(train_iter, train_loss_list, val_iter, val_loss_list, save_file)


if __name__ == "__main__":
    generate_result_from_log('D:\\graduationproject\\ver3\\log3.txt', 'D:\\graduationproject\\ver3\\result3.png')
