# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 6-25-2017
# Author: Alex H Chao
# ************************************************************************************

import heapq
import math as m
import numpy as np
from helper.helper import *
from random import randint

# 10.5 Compute running median
# ************************************************************************************
# TRICK: maintain 2 heaps, a min and max heap
# this passes all hacker rank tests

"""
min_heap = []
max_heap = []

min_heap = [randint(0,99) for x in range(20)]
hq.heapify(min_heap)
hq.nlargest(5,min_heap)
hq.nsmallest(5,min_heap)
"""
# https://www.hackerrank.com/challenges/find-the-running-median/copy-from/32447576
# Hacker rank test cases
"""
12
4
5
3
8
7
"""

class running_median(object):

    def __init__(self):
        self.min_heap, self.max_heap = [], []
        self._median = None

    def push(self, p):
        #if max_heap is empty
        if len(self.min_heap) == 0:
            heapq.heappush(self.min_heap, p)
        #compare vs median
        elif p < self.min_heap[0]: # if value <= max_heap[0], put in max heap
            heapq.heappush(self.max_heap, -p)
        else:
            heapq.heappush(self.min_heap, p)
        self.rebalance() # rebalance heaps to make sure they are "balanced"

    def rebalance(self):
        #if max_heap too big
        #print('rebalancing heaps...')
        if len(self.min_heap) < len(self.max_heap): # pass max_heap item to min heap
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap) +1:
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    @property
    def median(self):
        #print("using the getter")
        if len(self.min_heap) == len(self.max_heap): # if equal take avg
            return (self.min_heap[0] - self.max_heap[0])*0.5
        else:
            return self.min_heap[0]*1.0

    @median.setter
    def median(self,value):
        print('using the setter')
        self._median = value

    def __str__(self):
        return self.max_heap.extend(self.min_heap)


if __name__ == '__main__':

    m = running_median()

    for num in [12, 4, 5, 3, 8, 7]:
        m.push(num)
        #print(num)
        #print(m.max_heap, m.min_heap)
        print('running median = {}'.format(m.median))

