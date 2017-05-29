# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-29-2017 - Mon
# Author: Alex H Chao
# CH 11 - Searching
# ************************************************************************************
import math as m
import numpy as np
from helper.helper import *

# ************************************************************************************
# 11.8 - Find the k-th largest element (Quick select)
# assume array is unsorted and all entries are distinct
# same as partition function in quicksort() algo
# ************************************************************************************


def partition(a):
    """
    
    Parameters
    ----------
    a - array

    Returns
    -------
    array where [0:pivot] is less than pivot, [pivot+1:n] are greater than pivot
    """

    #set partition to be last element
    p = len(a)-1
    hi = p-1
    lo = 0

    while lo < hi:
        if (a[lo] > p) & (a[hi] < p): #we need to swap
            swap(a,lo,hi)
        elif a[lo] > p: # only lo is greater than p
            hi -= 1
        elif a[hi] < p:
            lo += 1
        #elif (a[lo] < p) & (a[hi] > p): #we good
        else:
            lo += 1
            hi -= 1

    #swap out the pivot
    swap(a,hi,p)



a = [9,9,9,9,1,5]

partition(a)
print(a)

