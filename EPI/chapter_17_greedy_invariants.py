# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-21-2017
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np
from random import randint


# ************************************************************************************
# 17.4 - The three-sum (3sum) problem (assume unsorted)
# ************************************************************************************
# 8/1 Tuesday - Argo tea

# leverage the 2sum function
def test_three_sum():
    a = [11,2,5,7,3]
    a.sort()
    print(two_sum(a,22))
    print(three_sum(a, 22))

# assume sorted
def two_sum(a, num):
    # 2 ptrs
    start = 0
    end = len(a)-1

    while start < end: # tricky, make sure its < not <=
        this_sum = a[start] + a[end]
        if this_sum == num:
            #we found it
            return True
        elif this_sum < num:
            start += 1
        else:
            end -= 1
    return False


def three_sum(a, target):
    # check if two_sum = target - current num
    a.sort()
    for x in a:
        found = two_sum(a, target - x)
        print(x, target-x)
        if found:
            return True
    return False



# ************************************************************************************
# 17.7 - Maximum Water Trapped by a Pair of Vertical Lines
# ************************************************************************************
# 8/1 Tuesday - Argo tea

# Methodology
# start with 2 pts -> 0, n-1
# calculate water trapped, kepping track of running max
# move ptrs inward (whichever is the min of the 2 gets moved)


a = [1,2,1,3,4,4,4,1]

start = 0
end = len(a)-1

running_max = 0

while start < end:
    water_trapped = min(a[start], a[end]) * (end-start)
    running_max = max( running_max, water_trapped )
    print(start, end, running_max)
    if a[start] <= a[end]:
        start += 1
    else:
        end -= 1




