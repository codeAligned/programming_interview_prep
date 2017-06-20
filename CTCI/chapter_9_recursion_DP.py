# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Cracking the Coding Interview 5th Edition
# Created 6-18-2017
# Author: Alex H Chao
# ************************************************************************************
import math
import numpy as np
import pandas as pd
from helper.helper import *
from random import randint

# ************************************************************************************
# 9.1 Staircase problem (1,2,3 steps)
# ************************************************************************************
#basically fibonacci
#slow version


def staircase(n):
    if n < 0:
        return 0
    elif n == 0:
        return 1
    else:
        return staircase(n-1) + staircase(n-2) + staircase(n-3)

# memoization
def staircase_dp(n, stairs):
    if n < 0:
        return 0
    elif n == 0:
        return 1
    else:
        if n in stairs.keys(): # just return it
            return stairs[n]
        else: #not in it, need to calculate it
            stairs[n] = staircase_dp(n-1, stairs) + staircase_dp(n-2, stairs) + staircase_dp(n-3, stairs)
            return stairs[n]

# === how much faster is memorization ?
def test_staircase():
    n=20
    d = {}
    staircase(n)
    staircase_dp(n,d)
    pd.DataFrame.from_dict(d,orient='index')




# ************************************************************************************
# 9.2 2-D Ways to Traverse - Robot problem - how many paths to go from (0,0) to (X,Y) ?
# ************************************************************************************
# base case is we hit a corner (e.g. x <= 0 or y <= 0) return 1
# (0,0) -> 0
# (x, y) = (X-1, Y) + (X, Y-1)

# Runtime: niave algo -> O(2 ^ X+Y) vs DP algo = O(X*Y)

@timeit
def traverse_2d_array(m,x,y):
    if x == y == 0:
        return 0
    elif x <= 0 or y <= 0:
        return 1
    else:
        m[x,y] = traverse_2d_array(m,x-1,y) + traverse_2d_array(m, x,y-1)


# no recursion method

def count_ways(m):
    num_rows = m.shape[0]
    num_cols = m.shape[1]
    m[0,:] = 1
    m[:,0] = 1

    for row in range(1,num_rows):
        for col in range(1,num_cols):
            m[row,col] = m[row-1,col] + m[row, col-1]
    return m

# this way (leetcode) doesnt involve numpy

def initialize_matrix_zeros(rows,cols):
    return [[0 for x in range(cols)] for y in range(rows)]

# working, submitted on leetcode

def count_ways(n,m):
    A = np.zeros([n,m])
    A[0, :] = 1
    A[:, 0] = 1

    for row in range(1, n):
        for col in range(1, m):
            A[row, col] = A[row - 1, col] + A[row, col - 1]
    return int(A[row,col])

def count_ways_2D_array(x,y):
    return math.factorial(x+y) / (math.factorial(x) * math.factorial(y))

n = 5
m = np.zeros([n, n])

n=3
m=4
count_ways(4,4)

count_ways_2D_array(3,3)

#random: u can do tuples in list comprehensions too

[(x, x+1) for x in range(4)]


# ************************************************************************************
# 9.3 Magic or Equilibrium Index of Array is such that A[i] = i
# Given sorted array, find a magic index, if any
# ************************************************************************************

# ************************************************************************************
# 9.3 Return all subsets of a set
# ************************************************************************************


# ************************************************************************************
# 9.4 - Compute all permutations of a string of unique chars
# ************************************************************************************


# ************************************************************************************
# 9.8 - Give infinite number of quarters, dimes, nickets, pennies, calculate how many
# ways of representing n cents

# ************************************************************************************


