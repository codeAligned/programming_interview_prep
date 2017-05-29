# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-21-2017
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np


# ************************************************************************************
# 16.1 - Count the number of score combinations of football game
# ************************************************************************************

# ************************************************************************************
# 16.2 - Compute the Levenshtein Distance (Minimum Edit Distance of two words)
# ************************************************************************************



# ************************************************************************************
# 16.3 - Count number of ways to traverse 2D Array (CTCI 9.2)
# ************************************************************************************
# answer should be ( X + Y ) ! / X! ! Y!

def count_ways_2D_array(x,y):
    return m.factorial(x+y) / (m.factorial(x) * m.factorial(y))

count_ways_2D_array(3,3)

def count_ways_2D_array_DP(x,y, num_ways):
    if x == y == 0:
        return 1
    if num_ways[x][y] == 0: # not set yet
        # if we've reached the edge, set to 0, else recurse
        top = 0 if y == 0 else count_ways_2D_array_DP(x, y-1, num_ways)
        left = 0 if x == 0 else count_ways_2D_array_DP(x-1,y,num_ways)
        num_ways[x][y] =  top + left
        print('setting {},{} = {}'.format(x, y,num_ways[x][y]))
    return num_ways[x][y]


n = 5
num_ways = np.zeros([n, n])
count_ways_2D_array_DP(n - 1, n - 1, num_ways)
num_ways


# ************************************************************************************
# 16.10 - Count the number of moves to climb stairs
# ************************************************************************************

# ************************************************************************************
# 16.6 - The Knapsack Problem
# ************************************************************************************

values = [60, 50, 70, 30]
weights = [5, 3, 4, 2]
capacity = 5

def knapsack(values,weights,capacity):
    V = np.zeros([len(values), capacity + 1])
    for i in range(len(values)):
        for w in range(capacity+1):
            # each new row is a new item
            # if the new item weights too much, we cant take it
            # else, we take the max of (take item) vs (dont take item)
            V[i][w] = V[i-1][w] if weights[i] > w else max(V[i-1][w - weights[i]] + values[i], V[i-1][w])
    return V

print(knapsack(values,weights,capacity))


# ************************************************************************************
# 16.8 Find Min Weight Path in a Triangle
# ************************************************************************************
# Trick - write as triangular matrix, work backwards


