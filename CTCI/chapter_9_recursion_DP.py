# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Cracking the Coding Interview 5th Edition
# Created 6-18-2017
# Author: Alex H Chao
# ************************************************************************************
import math as m
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


