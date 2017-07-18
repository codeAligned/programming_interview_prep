# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Dynamic Programming for Coding Interviews
# Created 6-24-2017
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


# ************************************************************************************
# Ch 9 Practice Questions - Edit Distance - p 78
# given 2 strings, find the min number of operations required to convert one string to
# the other (one operation = interset, remove, or replace
# ************************************************************************************
# WORKING - submitted on leetcode - https://leetcode.com/problems/minimum-path-sum/#/description


s1 = 'sunday'
s2 = 'saturday'

edit_distance(s1,s2,False)

edit_distance('a','',True)

def edit_distance(s1,s2, verbose=True):
    """
    Given 2 strings, compute the edit distance of the 2 strings
    Parameters
    ----------
    s1
    s2

    Returns
    -------

    """
    s1 = ' ' + s1 # add space to begining
    s2 = ' ' + s2
    A = initialize_matrix_zeros(len(s1), len(s2))

    rows = len(s1)
    cols = len(s2)

    A[0][0] = 0
    #fill out first row and col 1st
    for i in range(1,rows):
        A[i][0] = A[i-1][0] + 1 # copy sum down

    for j in range(1,cols):
        A[0][j] = A[0][j-1] + 1

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            #TRICK: if no edit is needed, just copy it over
            if s1[row] == s2[col]:
                A[row][col] = A[row-1][col-1]
            else: # we need to either replace, add, or remove
                A[row][col] = 1 + min(A[row-1][col],
                                      A[row][col-1],
                                      A[row-1][col-1])

    if verbose:
        return A
    else:
        return A[rows-1][cols-1]


from helper import *

# ************************************************************************************
# Ch 6 - Find length of longest substring such that sum of digits in first half
# equals the sum of digits in second half
# p 51
# example:
# input = '142124', output = 6
# input = '9430723', output = 4 (from 4307)

# ************************************************************************************
# 7-17-2017
# Argo tea - 7:51 PM

# Methodology
# M[i][j] = sum of digits from i through j
# 1) populate matrix - sum[i][j] -> sum of digits from i thru j (upper triagular matrix)
# 2) check every element with its "complement", e.g. (1,2) -> (3,4), (0,3) -> (4,7)
a = '142124'
list(a)
a = [9,4,3,0,7,2,3]
b = convert_list_chars_to_int(list(a))

n = len(b)
M = initialize_matrix_zeros(n,n)

for row in range(len)

# ************************************************************************************
# 9.3 String Interleaving, p88
# ************************************************************************************


# ************************************************************************************
# 9.4 Subset Sum, p 96
# ************************************************************************************


# ************************************************************************************
# 9.5 Longest Common Subsequence, p 100
# ************************************************************************************



# ************************************************************************************
# 9.8 Cutting a Rod, p 114
# ************************************************************************************




# ************************************************************************************
# 9.10 Longest Palindromatic Subsequence, p 121
# ************************************************************************************









