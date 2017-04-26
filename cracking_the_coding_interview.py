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
s = 'aabbcdef'

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

out

# 1.6 Given a N by N matrix, rotate it by 90 degrees
# ************************************************************************************




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


