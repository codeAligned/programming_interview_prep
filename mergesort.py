# ************************************************************************************
# INTERVIEW PREP -
# Sorting algos - Merge Sort
# Implement Has map
# Created 6-5-2017 - Sunday
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np
from helper.helper import *
from random import randint

# This is working - as of 6-5-2017


# merge function -> merge 2 sorted arrays
def merge(left, right):
    merged = []
    i, j = 0, 0
    size_left, size_right = len(left), len(right)
    while (i < size_left) & ( j < size_right):
        if left[i] <= right[j]: # iterate through both arrays, adding smaller value
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    while i < size_left: # add all left over items from the other array
        merged.append(left[i])
        i += 1
    while j < size_right:
        merged.append(right[j])
        j += 1
    print('merging {} and {} = > {} '.format(left, right, merged))
    return merged


def mergeSort(a):
    if len(a) <= 1:
        return a
    else:
        mid = int((len(a)) / 2 ) # trick, len(a) / 2 not len(a)-1 / 2!!
        print('mid = {}'.format(mid))
        left = mergeSort(a[:mid])
        right = mergeSort(a[mid:])
        return merge(left, right)

# ===========

a = [1,3]
b = [2,4,6,8]
merge(a,b)


a = [randint(1,9) for x in range(8)]
print(a)
mergeSort(a)


