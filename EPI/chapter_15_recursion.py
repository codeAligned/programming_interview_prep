# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-21-2017
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np
import pandas as pd
from helper.helper import *
from random import randint

# ************************************************************************************
# 15.1 - Towers of Hanoi
# ************************************************************************************


# ************************************************************************************
# 15.2 - Generate All Nonattacking Placements of n-Queens
# ************************************************************************************

# ************************************************************************************
# 15.3 - Generate Permutations
# ************************************************************************************

# basically if you have perm(n-1), to get perm(n) u just have to stick n in each spot
# inside each char of perm(n)

def perm(arr):
    if len(arr) <= 1:
        return arr

    for i in range(n):
        sub_arr = arr[:-1]
        n = arr[-1]


# side tangent ->
# Shuffle an array




# ================================
# SCRAP
# ================================

arr = [1,2,3]
shuffle(arr)
print(arr)

#is this right? lets run a simulation and check

N=6000
n = 3

d = {}

for i in range(N):
    arr = [x for x in range(1,n+1)]
    shuffle(arr)
    #print('adding {}'.format(arr))
    add_to_dict(d,str(arr))

print(convert_dict_to_dataframe(d))


def shuffle(arr):
    """
    Shuffle an array
    
    Parameters
    ----------
    arr - array

    Returns
    -------
    None
    
    """
    for i in range(len(arr)-1):
        #print(i)
        swap(arr,i, randint(i,len(arr)-1))



def add_to_dict(d, key):
    """
    Add sequence counts to dict 
    """
    if key in d.keys():
        d[key] += 1
    else:
        d[key] = 1


"""

class dict_of_counts(object):
    def __init__(self):
        self.d = {}

    def add(self, obj):
        if obj not in self.d.keys():
            self.d[obj] = 1
        else: # already exists
            self.d[obj] += 1

    def __repr__(self):
        return [i,v for i,v in self.d.items()]
"""