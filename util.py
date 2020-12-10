"""
created on 2020/10/28 12:10
@author:yuka
@note:catch exception
"""

import math


def match_data(x, f):
    return len(x) == len(f)


# todo test
def zero_data(x, f):
    # case 1 : zero
    case1 = (x == 0).all() | (f == 0).all()
    # case 2 : close to zero
    case2 = abs(max(f) - min(f)) < 1
    return case1 or case2


def nan_data(x):
    return isinstance(x, float) and math.isnan(x)
