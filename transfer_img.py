"""
created on 2020/12/11 11:37
@author:Shar
@note:
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import os.path as osp
import util
import cv2 as cv
import pandas as pd


def get_index(data_org, n=15):
    k = len(data_org)
    list1 = []
    list2 = []
    list1.append(2 * data_org.sum() / k)
    list2.append(0)
    for i in range(1, n):
        total1 = 0
        total2 = 0
        for j in range(k):
            tmp1 = data_org[j] * math.cos(2 * i * math.pi * j / k)
            tmp2 = data_org[j] * math.sin(2 * i * math.pi * j / k)
            total1 = total1 + tmp1
            total2 = total2 + tmp2
        total1 = 2 * total1 / k
        total2 = 2 * total2 / k
        list1.append(total1)
        list2.append(total2)
    return list1, list2


class Transfer:
    def __init__(self, viscosity, density, cs_area, diameter1, diameter2, length, omega, cycle, elasticity):
        self.v = viscosity
        self.d = density
        self.s = cs_area
        self.d1 = diameter1
        self.d2 = diameter2
        self.L = length
        self.w = omega
        self.T = cycle
        self.E = elasticity

    def get_a(self):
        a = (self.E / self.d) ** 0.5
        print(a)
        return a

    def get_c(self):
        m = self.d1 / self.d2
        b1 = (m * m - 1) / (2 * math.log(m)) - 1
        b2 = m ** 4 - 1 - ((m * m - 1) ** 2) / math.log(m)
        c1 = (2 * math.pi * self.v) / (self.d * self.s)
        tmp = self.w * self.L / self.s
        c2 = b1 + 2 / (tmp / math.sin(tmp) + math.cos(tmp))
        c = c1 * (1 / math.log(m) + 4 / b2 * (b1 + 1) * c2)
        print(c)
        return c

    def get_para1(self, a, c, n=15):
        list_alpha = []
        list_beta = []
        list_alpha.append(0)
        list_beta.append(0)
        for id_x in range(1, n):
            m1 = id_x * self.w / (a * math.sqrt(2))
            save = math.sqrt(1 + (c / (id_x * self.w)) ** 2)
            alpha_tmp = m1 * math.sqrt(1 + save)
            beta_tmp = m1 * math.sqrt(-1 + save)
            list_alpha.append(alpha_tmp)
            list_beta.append(beta_tmp)
        return list_alpha, list_beta

    def get_para2(self, alpha, beta, list1, list2, n=15):
        k = []
        miu = []
        k.append(0)
        miu.append(0)
        for i in range(1, n):
            p1 = list1[i] * alpha[i]
            p2 = list2[i] * beta[i]
            p3 = self.E * self.s * (alpha[i] ** 2 + beta[i] ** 2)
            tmp_k = (p1 + p2) / p3
            tmp_miu = (p1 - p2) / p3
            k.append(tmp_k)
            miu.append(tmp_miu)
        return k, miu

    def get_para3(self, k, miu, alpha, beta, list3, list4, n=15):
        O = []
        P = []
        x = self.L
        for i in range(n):
            p1 = math.cosh(beta[i] * x)
            p2 = math.sinh(beta[i] * x)
            p3 = math.sin(alpha[i] * x)
            p4 = math.cos(alpha[i] * x)
            O_tmp = (k[i] * p1 + list4[i] * p2) * p3 + (miu[i] * p2 + list3[i] * p1) * p4
            P_tmp = (k[i] * p2 + list4[i] * p1) * p4 - (miu[i] * p1 + list3[i] * p2) * p3
            O.append(O_tmp)
            P.append(P_tmp)
        return O, P

    def get_para4(self, alpha, beta, list1, list2, list3, list4, n=15):
        dO = []
        dP = []
        x = self.L
        for i in range(n):
            p1 = math.cosh(beta[i] * x)
            p2 = math.sinh(beta[i] * x)
            p3 = math.sin(alpha[i] * x)
            p4 = math.cos(alpha[i] * x)
            p5 = list4[i] / (self.E * self.s)
            p6 = list3[i] / (self.E * self.s)
            dO_tmp = ((beta[i] * list4[i] - alpha[i] * list1[i]) * p1 + p5 * p2) * p3 + (
                    (beta[i] * list1[i] + alpha[i] * list2[i]) * p2 + p6 * p1) * p4
            dP_tmp = ((beta[i] * list4[i] - alpha[i] * list1[i]) * p2 + p5 * p1) * p4 - (
                    (beta[i] * list1[i] + alpha[i] * list2[i]) * p1 + p6 * p2) * p3
            dO.append(dO_tmp)
            dP.append(dP_tmp)
        return dO, dP

    def get_result(self, O, P, dO, dP, n=15):
        u = []
        f = []
        list_t = np.linspace(0, self.T, 200)
        base1 = (O[0] + P[0]) / 2
        base2 = (dO[0] + dP[0]) / 2
        tmp = self.E * self.s
        for t in list_t:
            u_tmp = base1
            f_tmp = base2 * tmp
            for i in range(1, n):
                u_tmp = u_tmp + (O[i] * math.cos(i * self.w * t) + P[i] * math.sin(i * self.w * t))
                f_tmp = f_tmp + tmp * (dO[i] * math.cos(i * self.w * t) + dP[i] * math.sin(i * self.w * t))
            u.append(u_tmp)
            f.append(f_tmp)
        return u, f


if __name__ == '__main__':
    # filepath = 'D:\\data\\orgData\\FD-7.csv'
    # file = pd.read_csv(filepath, usecols=['位移', '载荷'], encoding='ANSI')
    # maxlength = len(file)
    # exception_zero = 0
    # pre = osp.splitext(osp.basename(filepath))[0]
    # f_max = max(np.array(file)[:, 1])
    # print(pre + ' start, max_f = ' + str(f_max))
    # # split
    # cnt = 0
    # idx = 0
    # # image_cnt = 0
    # while maxlength - 200 * cnt > 0:
    #     # while cnt < 5:
    #     temp = file[idx:idx + 200]
    #     data = np.array(temp)
    #     x_array = data[:, 0]
    #     f_array = data[:, 1]
    #     idx += 200
    #     cnt += 1
    #     # skip exception
    #     if util.zero_data(x_array, f_array):
    #         exception_zero += 1
    #         # print(pre + '-' + str(cnt) + ':zero')
    #         continue

    filepath = 'D:\\data\\orgData\\tp7.xlsx'
    data = pd.read_excel(filepath).iloc[:, [8, 11]].values
    pre = osp.splitext(osp.basename(filepath))[0]
    image_cnt = 0
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
        mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
        mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
        plt.plot(x_array, f_array)
        plt.title('地面示功图')
        plt.show()
        cv.waitKey()
        ts = Transfer(0.5, 6461, 4.9e-4, 0.06, 0.025, 300, 0.445, 14.12, 2.058e11)
        a = ts.get_a()
        c = ts.get_c()
        list1, list2 = get_index(f_array)
        list3, list4 = get_index(x_array)
        alpha, beta = ts.get_para1(a, c)
        k, miu = ts.get_para2(alpha, beta, list1, list2)
        O, P = ts.get_para3(k, miu, alpha, beta, list3, list4)
        dO, dP = ts.get_para4(alpha, beta, list1, list2, list3, list4)
        u, f = ts.get_result(O, P, dO, dP)
        plt.plot(u, f)
        plt.title('泵功图')
        plt.show()
        cv.waitKey()
