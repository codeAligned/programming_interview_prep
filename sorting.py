# ************************************************************************************
# INTERVIEW PREP -
# Sorting algos - Quick Sort and Merge Sort
# Implement Has map
# Created 5-30-2017 - Tues
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np
from helper.helper import *
from random import randint

# not working!!

def partition(a, lo, hi):
    """

    Parameters
    ----------
    a - array

    Returns
    -------
    array where [0:pivot] is less than pivot, [pivot+1:n] are greater than pivot
    """

    # set partition to be last element
    p = len(a ) -1
    hi = p- 1 if p is None else hi
    lo = 0 if lo is None else lo

    while lo < hi:
        if (a[lo] > a[p]) & (a[hi] < a[p]):  # we need to swap
            swap(a, lo, hi)
        elif a[lo] > a[p]:  # only lo is greater than p
            hi -= 1
        elif a[hi] < a[p]:
            lo += 1
        # elif (a[lo] < p) & (a[hi] > p): #we good
        else:
            lo += 1
            hi -= 1

    # swap out the pivot
    new_p = hi if a[hi] > a[p] else hi + 1

    swap(a, new_p, p)
    return new_p


def quicksort(a,start,end):
    if start == end:
        return a[start]
    if start < end:
        p = partition(a,start,end)
        left = quicksort(a,start,p)
        right = quicksort(a,p+1, end)

    return left + p + right


a = [randint(1,9) for x in range (10)]


quicksort(a,0,len(a)-1)

