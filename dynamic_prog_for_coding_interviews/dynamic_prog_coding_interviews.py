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
# Example 8.1 - Minimum Cost Path - p 62
# given n x m matrix - calculate min path from (0,0) to (n,m)
# ************************************************************************************
# WORKS - submitted on leetcode - https://leetcode.com/problems/minimum-path-sum/#/description

M = [[1,3,5,8],[4,2,1,7],[4,3,2,3]]
len(M[0])
len(M)
calc_min_cost_path(M)

def calc_min_cost_path(M):
    """
    input an m by n matrix M of costs,
    return the min cost from (0,0) to (m,n)
    """
    A = initialize_matrix_zeros(3, 4)

    rows = len(M)
    cols = len(M[0])

    A[0][0] = M[0][0]
    #fill out first row and col 1st
    for i in range(1,rows):
        A[i][0] = A[i-1][0] + M[i][0] # copy sum down

    for j in range(1,cols):
        A[0][j] = M[0][j] + A[0][j-1]

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            A[row][col] = M[row][col] + min(A[row-1][col], A[row][col-1])
    return A[rows-1][cols-1]