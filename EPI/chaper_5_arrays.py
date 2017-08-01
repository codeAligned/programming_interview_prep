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


# ************************************************************************************
# 5.5 - Delete dupes from a sorted array
# ************************************************************************************


# ************************************************************************************
# 5.6 - Buy and Sell a stock once for max profit (opposite of max drawdown)
# ************************************************************************************

a = [randint(0,9) for x in range(10)]
a
max_profit(a)

# min so far
# max so far
# max ending here

def max_profit(a):
    MIN_NUM = -999
    MAX_NUM = 999

    running_min = MAX_NUM
    running_max = MIN_NUM
    max_ending_here = MIN_NUM
    max_so_far = MIN_NUM

    for i, num in enumerate(a):
        if num < running_min:
            running_min = num # new running min
            running_max = num # reset running max

        running_max = max(running_max, num) # update running max if needed
        max_ending_here = running_max -running_min
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

# solution

# ************************************************************************************
# 5.7 - Buy and Sell a stock Twice for max profit (the second buy must be made on another
# date after the first sale)
# ************************************************************************************


# ************************************************************************************
# 5.8 - Compute an alternation - write program that takes array A of numbers, rearranges
# A so that A[0] <= A[1] >= A[2] <= A[3] >= A[4]
# ************************************************************************************
a = [randint(0, 9) for x in range(10)]


# ************************************************************************************
# 5.9 - Enumerate all primes to n
# ************************************************************************************
# repeat of CTCI

def enumerate_primes_to_n(n):
    bool_arr = [1 for x in range(n)]
    # set first 2 index to be 0
    bool_arr[:2]=[0,0]
    primes = []
    for i in range(2, int(n**0.5) +1): # only need to go up to sqrt(2)
        for j in range(2*i,n,i): # seive all multiples
            bool_arr[j] = 0
    for i,v in enumerate(bool_arr):
        if v== 1:
            primes.append(i)
    return primes

enumerate_primes_to_n(100)

# ************************************************************************************
# 5.10 - Permute elements of an array - Given array A, and permutation array P,
# apply P to A
# ************************************************************************************

# ************************************************************************************
# 5.10 - Compute the next permutation
# ************************************************************************************

# ************************************************************************************
# 5.14 - Compute random permutation (shuffle)
# ************************************************************************************

# for every i
# pick random number from i+1 to n
# swap ( i, random num)
a = [x for x in range(10)]

def permute(a):
    n= len(a)-1
    for i in range(0,n-1):
        swap(a,i, randint(i+1,n))

# Test it out
# ============

d = {}
for i in range(10000):
    a = [x for x in range(1,11)]
    permute(a)
    add_to_dict(d, a[0])
    print('Adding {} to dictionary'.format(a[0]))
d


def add_to_dict(d, key):
    """
    Add sequence counts to dict 
    """
    if key in d.keys():
        d[key] += 1
    else:
        d[key] = 1


# ************************************************************************************
# 5.18 - Print the spiral ordering of a 2D array (clockwise, starting at 0,0)
# ************************************************************************************

n=4
x = [[i+j*(n) for i in range(n)] for j in range(n)]

"""
Row                     | Col
------------------------|-------------------------             
 first + i              | (first + i , last - i)
(first + i, last - i)   |  last - i
 last - i               | (last - i, first + i)
(last - i, first + (i+1)| first + i

"""
n
for i in range(n/2): # what if its odd?
    print('i=',i)
    print(print_ith_layer(x,i))
# IN PROGRESS


def print_ith_layer(x, i):
    """
    
    Parameters
    ----------
    x -> n x n 2D array
    i -> layer in which to print in spiral format

    Returns
    -------
    None
    """
    #import pdb; pdb.set_trace()
    first = 0
    last = len(x)-1
    out = []
    top = flatten_list(x[ first + i , first + i : last - i ].tolist()) #top
    right = flatten_list(x[ first + i : last - i +1  , last - i].tolist() )# right
    bottom = flatten_list(x[ last - i ,  first + i: last - i ].tolist() )# bottom
    left = flatten_list(x[ first + i + 1: last - i , first + i].tolist() )# left

    bottom = bottom[::-1]
    left = left[::-1]

    for l in [top,right,bottom,left]:
        out.extend(l)
    return out

def flatten_list(l):
    """
    
    Parameters
    ----------
    l - list of lists, e.g. [[0,1,2]]

    Returns
    -------
    list - [0,1,2]
    """
    return reduce(lambda x,y: x+y, l)


print(x[ first + i , first + i : last - i ]) # top
print(x[ first + i : last - i +1  , last - i]) # right
print(x[ last - i ,  first + i: last - i ]) # bottom
print(x[ first + i + 1: last - i +1 , first + i]) # left


x[first:last+1, last]

x = np.matrix(x)
s = x[0:3,1]
s[::-1]

x[3,:]


# ************************************************************************************
# 5.19 - Rotate a 2D array (CTCI)
# ************************************************************************************
