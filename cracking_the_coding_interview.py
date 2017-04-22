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

# 10:31 PM
dfdfdf


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


