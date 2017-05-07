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



# ************************************************************************************
# 5.1 - Dutch National Flag Problem - you have array A, and pivot index i, write function
# to arrange numbers so that all numbers < pivot is first, followed by =, followed by greater
# ************************************************************************************
#simular to quickselect algorithm except quick select is only 2 buckets [ <= p] [ > p ]

# ************************************************************************************
# Partition algorithm from quick sort
# ************************************************************************************
a = [randint(1, 3) for x in range(10)]

def partition(a, p):
    """
    reorders so that <p comes first, followed by >p
    a: array
    p = index (pivot)
    """
    start = 0
    end = len(a)-1
    pivot = a[p]
    while start < end:
        if a[start] <= pivot:
            start += 1
        else: # a[start] > pivot
            print('swapping {} and {}'.format(a[start], a[end]))
            swap(a,start,end)
            print(a)
            end -= 1
    return a

# ************************************************************************************
# now, lets add an EQUALS section
# Here is a good visualization of it
# http://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/

# here's the trick, have 3 ptrs situated like this:

# [ 0 0 0 1 1 1 ? ? ? 2 2 2]
#         ^     ^   ^
#         |     |   |
#         lo    mid hi

# let p == 1
# case 0: a[mid] < 1
#             swap(lo,mid), lo++, mid++
# case 1: a[mid] == 1
#             mid++
# case 2: a[mid] > 1
#             swap(mid, hi), hi --

a = [randint(0,9) for x in range(10)]
a
dutch_flag(a, 3)

def dutch_flag(a, p):
    lo = 0
    mid = 0
    hi = len(a)-1

    pivot = a[p]
    while mid < hi:
        if a[mid] < pivot:
            swap(a,lo,mid)
            lo += 1
            mid += 1
        elif a[mid] == pivot:
            mid += 1
        else: # > pivot
            swap(a,mid,hi)
            hi -= 1
    return a




