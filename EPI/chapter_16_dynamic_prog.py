# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-21-2017
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np

# ************************************************************************************
# 16.3 - Count number of ways to traverse 2D Array (CTCI 9.2)
# ************************************************************************************
# answer should be ( X + Y ) ! / X! ! Y!

def count_ways_2D_array(x,y):
    return m.factorial(x+y) / (m.factorial(x) * m.factorial(y))

count_ways_2D_array(3,3)

def count_ways_2D_array_DP(x,y, num_ways):
    if x == y == 0:
        return 1
    if num_ways[x][y] == 0: # not set yet
        # if we've reached the edge, set to 0, else recurse
        top = 0 if y == 0 else count_ways_2D_array_DP(x, y-1, num_ways)
        left = 0 if x == 0 else count_ways_2D_array_DP(x-1,y,num_ways)
        num_ways[x][y] =  top + left
        print('setting {},{} = {}'.format(x, y,num_ways[x][y]))
    return num_ways[x][y]


n = 5
num_ways = np.zeros([n, n])
count_ways_2D_array_DP(n - 1, n - 1, num_ways)
num_ways
