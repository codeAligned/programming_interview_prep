# -*- coding: utf-8 -*-
"""
Cracking the Coding Interview
Updated 11/26/2016
@author: alex
"""

# ************************************************************************************
# ch 1 - Array and Strings
# ************************************************************************************
# 1.1 Implement an algo to determine if a string has all unique characters (you cannot use any data structures)
# ************************************************************************************

# 10:31 PM
ord('a')
ord('0')

def are_all_chars_unique(s):
    # ascii boolean array
    arr = np.zeros(128)
    for c in s:
        if arr[ord(c)] == 1:
            print('{} is a dupe!'.format(c))
            return False
        else:
            arr[ord(c)] = 1
    return True

# 1.2 Write a function that reverses a string
# ************************************************************************************
# we can use a list as a stack

def reverse_string(s):
    p1 = 0 # 2 pointers, 1 starting from left, other from the right
    p2 = len(s)-1
    while p2 > p1:
        swap(s,p1,p2)
        p1 += 1
        p2 -= 1

def swap(arr,i,j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

arr = [1,2,3,4,5]
reverse_string(arr)
arr

swap(arr,0,2)

# 1.3 Write a function that checks if one string is a permutation of the other
# ************************************************************************************
# use a dict, collect word -> counts, traverse s1
# traverse s2, decrement
#11:07


def are_strings_permutations(s1,s2):
    d = {}
    for c in s1:
        if c not in d.keys():
            d[c] = 1
        else:
            d[c] += 1

    for c in s2:
        if c not in d.keys():
            return False
        elif d[c] <= 0:
            return False
        else:
            d[c] -= 1
    return True

# Runtime = O(n + m)

s1 = 'abbcda'
s2 = 'dcbbaa'

are_strings_permutations(s1,s2)

def get_word_counts(s):
    d = {}
    for c in s:
        if c not in d.keys():
            d[c] = 1
        else:
            d[c] += 1
    return d



# 1.5 Write a function that compresses a string so aabcccccaaa is a2b1c5a3
# ************************************************************************************
# use tuples?
#11:24
s = 'aabbbbbbbcdef'

def compress(s):
    count = 1
    out = []
    last_char = s[0]

    for j in range(1,len(s)):
        if s[j]==last_char:
            count += 1
        else:
            out.append(last_char)
            out.append(count)
            count = 1
            last_char = s[j]

    out.append(last_char)
    out.append(count)

    if len(out) >= len(s):
        return s
    return ''.join([str(x) for x in out])


compress(s)


# 1.6 Given a N by N matrix, rotate it by 90 degrees
# ************************************************************************************
# 5-2-17 Tues
n = 5
import numpy as np

x = [[i+j*(n) for i in range(n)] for j in range(n)]

def initialize_matrix(n):
    return [[i+j*(n) for i in range(n)] for j in range(n)]

n=9

#lets do outer layer first
def rotate_matrix(m):
    n = len(m)
    first = 0
    for first in range(n-1):
        last = n-1-first

        for offset in range(first, last):
            #print(offset)
            temp = m[first][offset]
            m[first][offset] = m[offset][last]
            m[offset][last] = m[last][last - offset]
            m[last][last - offset] = m[last - offset][first]
            m[last - offset][ first] = temp
            #print(first,offset, ' <- ', offset, last)
        #print('first = ',first)
        #print(np.matrix(m))
        return m

m = initialize_matrix(3)
m = np.arange(9).reshape([3,3])
print(rotate_matrix(m))



# 1.7 Given a M by N matrix, if an element is 0, its entire row and column are set to 0
# ************************************************************************************
# 5-3-17 Wed

#algorithm - keep 2 boolean arrays, row and col
n=5
m = random_matrix(n,min=0,max=4)

rows = [0 for x in range(n)]
cols = [0 for x in range(n)]

# iterate through and mark the rows and cols arrays

for row in range(n):
    for col in range(n):
        if m[row][col] == 0:
            rows[row] = 1
            cols[col] = 1

#now iterate thru rows, cols and nullify each row/col

for row,value in enumerate(rows):
    if value:
        m[row][:] = [0 for x in range(n)]

for col,value in enumerate(cols):
    if value:
        m[:][col] = [0 for x in range(n)]

# in progress .......
# annoying, might have to revert to using numpy matrices


print(np.matrix(m))

rows
cols


# 1.8 given a isSubstring function, write a function to check whether two strings, s1, s2
# is a rotation of the other
#  ****************
# Here's the trick - if s1 -> xy, then s2 would have to be yx
# yx is always a substring of xyxy, hence s2 would have to be a substring of s1s1

s2 = 'waterbottle'
s1 = 'erbottlewat'

def are_strings_rotations(s1,s2):
    if s2 in s1+s1:
        return True
    else:
        return False

are_strings_rotations(s1, s2)



# ************************************************************************************
# ch 17 - Moderate
# ************************************************************************************

# 17.1 - Write a function to swap a number in place
# ************************************************************************************
# python doesnt swap in place
# |---------|---------------|---
# 0         b_0  (diff)    a_0

def swap(a, b):  # assume a > b
    a = a - b  # a = diff
    b = a + b  # b = diff + b = a
    a = b - a  # a = a- diff = b
    print a, b


arr = [1, 2, 3, 4]
swap(arr[0], arr[1])
print arr


# 17.3 - Write a function to compute number of trailing zeros in a factorial
# ************************************************************************************

def compute_trailing_zeros(n):
    total = 0
    for i in range(2, n + 1):
        total += count_multiples_of_five(i)
    return total


# count multiples of 5
def count_multiples_of_five(n):
    count = 0
    while (n % 5 == 0):
        count += 1
        n /= 5
    return count


compute_trailing_zeros(10)


def factorial(n):
    if n <= 1:
        return 1
    elif n == 2:
        return 2
    else:
        return n * factorial(n - 1)

# 17.4 - Write a function to compute max of two numbers (cant use if/else or any compares)
# ************************************************************************************


# 17.13 - Write a function to convert a binary search tree to a doubly linked list
# ************************************************************************************


# ************************************************************************************
# ch 18 - Hard
# ************************************************************************************


