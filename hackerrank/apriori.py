# A Priori Coding Test
# 6-25-2017
# Hacker Rank
# ============================

import heapq
import math as m
import numpy as np
from helper.helper import *
from random import randint


# minimum number of moves

#
a = [1234,4321]
b = [2345,3214]

[int(d) for d in str(n)]

total = 0

def minimum(a,b):
    total = 0
    for i, j in zip(a,b):
        i2 = [int(d) for d in str(i)]
        j2 = [int(d) for d in str(j)]
        print(i2,j2)
        for k in range(len(i2)):
            total += abs(i2[k] - j2[k])
    return total