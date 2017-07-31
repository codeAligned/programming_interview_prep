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


from helper.helper import *
import numpy as np

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
#a = [9,4,3,0,7,2,3]

# THIS WORKS
def equilibrium_array(a):
    a = convert_list_chars_to_int(list(a))

    n = len(a)
    #M = initialize_matrix_zeros(n,n)

    # base case - initialize diagonals first
    M = np.diag(a)

    # fill out the Matrix M
    for row in range(n):
        for col in range(row+1,n):
            M[row][col] = M[row][col-1] + a[col]

    # Now check each row and its complement, if they are equal,
    # keep track of running max
    running_max = 0
    for row in range(n):
        for col in range(row+1,int(n/2)): # note, we should stop at len/2!
            row_c, col_c = find_complement(row,col)
            if M[row][col] == M[row_c][col_c]:
                print('match found at: ({},{}) and ({},{})'.format(
                    row,col,row_c,col_c
                ))
                length = col-row+1
                if length > running_max:
                    print('updating new max to : {}'.format(length))
                    running_max = length

    return running_max


def find_complement(row,col):
    """
    helper function to find the "complement"
    """
    return col+1, col+1 + col-row


# ************************************************************************************
# 9.3 String Interleaving, p88
# ************************************************************************************


# ************************************************************************************
# 9.4 Subset Sum, p 96
# ************************************************************************************


# ************************************************************************************
# 9.5 Longest Common Subsequence, p 100
# Given 2 strings, retrusn total characters in their LCS
# Note, they dont need to be continuous
# ex) ABCD, AEBD, LCS = ABD, answer = 3
# ************************************************************************************
# 8:35 PM, argo tea, mon 7/17/17, solved 8:56

# Methodology
# 1) add null space at begining of each string
# 2) fill out Matrix M with following formula
#   if a[row] == b[col] # they are equal
#       M[row][col] = 1 + max( M[row-1][col], M[row][col-1] )
#    else :
#       M[row][col] = 1 + max( M[row-1][col], M[row][col-1] )

a, b = 'ABCD' , 'AEBD'
longest_common_subsequence(a,b)

def longest_common_subsequence(a,b):
    """
    Returns the longest common subsequence
    Parameters
    ----------
    a = string
    b = string

    Returns
    -------
    number (or Matrix)
    """
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    M = initialize_matrix_zeros(n+1,m+1)

    # remember to -1 to indexing of a,b

    # start at row, col = 1
    lcs = []
    for row in range(1,n+1):
        for col in range(1,m+1): # m+1 since we added a "null" col
            if a[row-1] == b[col-1]: # subtract one since we add null col
                #print(a[row-1],b[col-1])
                lcs.append(a[row-1])
                M[row][col] = 1 + max( M[row-1][col], M[row][col-1] )
                # letters equal, increment LCS
            else:
                M[row][col] = max(M[row - 1][col], M[row][col - 1])
                # else, carry over

    return M, lcs


# ************************************************************************************
# 9.8 Cutting a Rod, p 114
# ************************************************************************************
# 7-31

price = [1,5,8,9,10,17,17,20]
# from the book
# NEEDS REVIEW

max_values = [0 for x in range(len(price)+1)]

for i in range(1,len(price)+1):
    for j in range(i):
        max_values[i] = max(max_values[i], price[j] + max_values[i-j-1])

max_values

# ************************************************************************************
# 9.10 Longest Palindromatic Subsequence, p 121
# Given string, find length of longest palindrome sequence
# ************************************************************************************
# 9:01 PM - 7/17/17 - argo tea

a = 'BBABCBCAB'
b = 'abcdeffedcbc'
longest_palindromatic_subsequence(b)

def longest_palindromatic_subsequence(a):
    a = list(a)
    n = len(a)

    # initialize (diag are all ones, as a[i,i] will always be a palindrome length 1
    M = np.diag(np.ones(n))

    for k in range(1, n ): # k is just length from i to j
        for i in range(n - k ):
            j = i + k
            #print('i,j,k = {}, {}, {}'.format(i, j, k))
            # base case, if letters are same and adjecent, should be 2
            if (a[i] == a[j]) & (k==1):
                M[i][j] = 2
            elif a[i] == a[j]: # start and ends are equal -> recurse
                M[i][j] = 2 + M[i+1][j-1]
            else: # start and ends are not equal
                M[i][j] = max( M[i+1][j], M[i][j-1] )

    return M, M[0][n-1]



# ==== book gives weird loops, I made it simplier

n = 5
for k in range(2,n+1):
    for i in range(n-k+1):
        j = i+k-1
        print('i,j,k = {}, {}, {}'.format(i,j,k))

for k in range(1, n):  # k is just length from i to j
    for i in range(n - k):
        j = i + k
        print('i,j,k = {}, {}, {}'.format(i, j, k))

# ====


