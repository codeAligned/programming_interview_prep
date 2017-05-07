# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-7-2017
# Author: Alex H Chao
# ************************************************************************************


import numpy as np
from random import randint
from helper.helper import *


# ************************************************************************************
# Even Odds - Reorder an array so that its evens appear first
# ************************************************************************************
# 2 points, one from front, one from back

a = [randint(1, 9) for x in range(10)]
evens_and_odds(a)

def evens_and_odds(a):
    ptr_evens = 0
    ptr_odds = len(a)-1

    while ptr_evens < ptr_odds:
        if a[ptr_evens] % 2 == 0: #if even
            ptr_evens += 1 # go to the next line
        else: # its odd
            print('swapping {} and {}'.format(a[ptr_evens], a[ptr_odds]))
            swap(a,ptr_evens,ptr_odds)
            ptr_odds -= 1 # decrement odds
    return a




