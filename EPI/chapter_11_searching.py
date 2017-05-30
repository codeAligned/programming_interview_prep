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
from random import randint
# ************************************************************************************
# 11.8 - Find the k-th largest element (Quick select)
# assume array is unsorted and all entries are distinct
# same as partition function in quicksort() algo
# ************************************************************************************
# leet code -> https://leetcode.com/problems/kth-largest-element-in-an-array/#/solutions


# need to also pass in left, right, pivot_index
def partition(a, lo, hi):
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
    hi = p-1 if p is None else hi
    lo = 0 if lo is None else lo

    while lo < hi:
        if (a[lo] > a[p]) & (a[hi] < a[p]): #we need to swap
            swap(a,lo,hi)
        elif a[lo] > a[p]: # only lo is greater than p
            hi -= 1
        elif a[hi] < a[p]:
            lo += 1
        #elif (a[lo] < p) & (a[hi] > p): #we good
        else:
            lo += 1
            hi -= 1

    #swap out the pivot
    new_p = hi if a[hi] > a[p] else hi+1

    swap(a,new_p,p)
    return new_p


a = [randint(1,9) for x in range (10)]


k=5

# STILL IN PROGRESS

def get_kth_largest_element(a,lo,hi, k):

    p = partition(a,lo,hi)
    # kth largest = len(a) - p
    k_th_largest = len(a) - k
    if p == k_th_largest: # were done
        return a[p]
    elif p > k_th_largest: #
        get_kth_largest_element(a, 0, p, k)
    else: # p < k
        get_kth_largest_element(a, p+1, len(a)-1, k)

k=5
get_kth_largest_element(a,0,len(a)-1, 5)

partition(a)
print(a)

