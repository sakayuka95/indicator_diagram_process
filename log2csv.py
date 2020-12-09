"""
created on 2020/12/9 16:38
@author:Shar
@note:
"""
import csv
import os


def log2csv(log_path, flag):
    log_data = open(log_path, 'r')
    file_name = log_path.split(os.sep)[-1]
    save_file_name = file_name.replace('log', 'csv')
    csv_path = log_path.replace(file_name, save_file_name)
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for line in log_data:
        if flag == 0:
            if 'Train net output #' in line and 'accuracy =' in line:
                train_acc.append(float(line.split(' ')[-1]))
            elif 'Train net output #' in line and 'loss =' in line:
                train_loss.append(float(line.split(' ')[-2]))
            elif 'Test net output #' in line and 'accuracy =' in line:
                val_acc.append(float(-1))
                val_acc.append(float(line.split(' ')[-1]))
            elif 'Test net output #' in line and 'loss =' in line:
                val_loss.append(float(-1))
                val_loss.append(float(line.split(' ')[-2]))
        else:
            if 'Iteration' in line:
                train_acc.append(float(line.split(' ')[-1]))
                train_loss.append(float(-1))
                val_acc.append(float(-1))
                val_loss.append(float(-1))
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('filename', 'train_acc', 'train_loss', 'val_acc', 'val_loss'))
        # for i in range(len(train_acc)):
        #     tmp = [file_name, train_acc[i], train_loss[i], val_acc[i], val_loss[i]]
        #     writer.writerow(tmp)
        tmp = [file_name, train_acc, train_loss, val_acc, val_loss]
        writer.writerow(tmp)
    return csv_path


if __name__ == '__main__':
    # refs = ('train_acc', 'train_loss')
    # log2csv("C:\\Users\\42262\\Desktop\\train45.log", *refs)
    path1 = log2csv("C:\\Users\\42262\\Desktop\\train45.log", 0)
    path2 = log2csv("C:\\Users\\42262\\Desktop\\train.log", 1)
    print(path1)
    print(path2)
