# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Cracking the Coding Interview
# Hackerrank and Leetcode
# Author: Alex H Chao
# ************************************************************************************


import numpy as np
from random import randint

print(randint(0, 9))

# Elements of Programming Interviews
# ************************************************************************************
# To do Still:
# - Implement 3 sorting algos (mergesort, quicksort, etc)
# - Implement linked list and BST
# - Implement stack / queue
# ************************************************************************************

# Bitwise Ops

# USING XOR to single out non-pairs

arr = [1, 1, 2, 2, 1]

result = arr[0]
for i in range(1, len(arr)):
    result = result ^ arr[i]
    print result


# ************************************************************************************
# EPI - ch 6 - Arrays
# ************************************************************************************

# Example: given an array of integers, reorder so that its even entries appear first


# swap, pass by reference
def swap(a, i, j):
    """
    input: array a, swap indicies i,j 
    """
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
    return a


a = [x for x in range(1, 10)]
swap(a, 0, 1)
a  # a is changed!

# generate a random vector
a = [randint(1, 9) for x in range(10)]


# ************************************************************************************
# ch 6 - Arrays - From EPI ch6 Arrays, p 54,
# Given array, reorder so even entries appear first
# ************************************************************************************

# TRICK - DIVIDE INTO 3 SUBARRAYS - EVENS, UNCLASSIFIED, ODD
# ITERATE THROUGH ALL UNCLASSIFIED

# even = 0
def even_odds_2(a):
    ptr = 0
    odd = len(a) - 1
    while ptr < odd:
        if a[ptr] % 2 != 0:  # if its odd, swap
            swap(a, odd, ptr)  # decrement odd
            odd -= 1
        else:  # if even. move to next element
            ptr += 1
            # even += 1
    return a


even_odds_2([randint(1, 9) for x in range(10)])


def evens_odds(a):
    """
    From EPI ch6 Arrays, p 54, Given array, reorder so even entries appear first
    """
    # initialize 2 points, one at each end
    start_evens = 0
    end_odds = len(a) - 1
    while start_evens < end_odds:  # these 2 pointers will move inwards until they meet in middle
        # if its even, move on to the next one
        if a[start_evens] % 2 == 0:
            start_evens += 1
        else:  # otherwise, swap it out
            swap(a, start_evens, end_odds)
            # move odds pointer inwards
            end_odds -= 1
    return a


evens_odds([randint(1, 9) for x in range(10)])

from collections import deque

# rotate an array (Hacker rank)
# trick is to find a formula for the new position (k mod n)


s = "saveChangesToSpace"
if s is None:
    print 0
count = 1
for letter in s:
    if letter.lower() != letter:
        count += 1
print count


# matching socks



# Check if n is prime
# ************************************************************************************
def is_prime(n):
    for i in range(2, int(np.math.sqrt(n) + 1)):
        if n % i == 0:
            return 0
    return 1


print [x for x in range(20)]
print [is_prime(x) for x in range(20)]

# 8.6 - Write a program tha takes an integer n and returns all primes between 1 and n
# EPI p65
# ************************************************************************************
# tip: for each i, only go up to sqrt(i)
# tip: seive or remove all multiples of any primes from future calculations

# also known as Sieve of eratosthenes

N = 20


def generate_all_primes(N):
    # keep track of all primes
    is_prime_arr = [1 for i in range(N)]
    is_prime_arr[0] = is_prime_arr[1] = 0
    primes = []
    for n in range(2, N):  # start at 2
        # divide each i by 2 to srt(i)
        if is_prime_arr[n]:
            primes.append(n)
            # seive out multiples
            for j in range(n, N, n):
                is_prime_arr[j] = 0
    return primes


generate_all_primes(50)


# ************************************************************************************
# Write program that takes an array A and index i , and rearranges elements so that all
# elements < A[i] (the pivot) appear 1st, then elements == p, then elements > p
# INCOMPLETE, NEED TO REVISIT
# ************************************************************************************

def dutch_flag(p, s):
    """
    keep invariants:
    bottom group: [0, smaller-1]
    middle group: [smaller, equal-1]
    unclassified: [equal, larger-1]
    largergroup:  [larger, N]
    """
    smaller = 0
    larger = len(s) - 1  # initialize at end
    equal = 0  # this will be our "main" pointer
    pivot = s[p]  # initialize pivot

    while equal < larger:
        if s[equal] < pivot:
            # put in smaller group
            swap(s, equal, smaller)
            # increment both
            equal += 1
            smaller += 1
        elif s[equal] > pivot:
            # put in larger group
            swap(s, equal, larger)
            larger -= 1
            # equal += 1 # dont increment
        else:
            equal += 1
        print s
    return s

# WORKING (added on 5-7-2017)

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






# generate a random vector
a = [randint(1, 3) for x in range(10)]
a
p = 0
dutch_flag(p, a)

# ************************************************************************************
# EPI 6.2, p 59
# WRite a program that increments a decimal number from D to D+1, e.g. [1,2,9] -> [1,3,0]
# ************************************************************************************
a = [1, 8, 8]
for i in reversed(range(len(a))):
    if a[i] == 9:
        a[i] = 0
    else:
        a[i] += 1
        break
print a


# ************************************************************************************
# EPI 6.6, p 62
# Write program that takes an array of daily stock prices and returns max profit of selling
# 1 share
# ************************************************************************************

def max_profit(price):
    max_so_far = 0
    min_price_thus_far = 9999

    for i in range(len(price)):
        min_price_thus_far = min(min_price_thus_far, price[i])
        profit_if_sold_today = price[i] - min_price_thus_far
        max_so_far = max(max_so_far, profit_if_sold_today)
    return max_so_far


price = [randint(80, 100) for x in range(10)]
price
max_profit(price)

# ************************************************************************************
# Cracking the Coding Interview (CTCI) - Ch 17 - Moderate
# 17.8 - Given array of integers (both positive and negative) find contiguous max subarray with largest sum
# WARNING: DOESNT SUPPORT ARRAYS WITH ALL NEGATIVE NUMBERS OR EMPTY ARRAYS (RETURNS 0)
# ************************************************************************************

arr = [2, 3, -8, -1, 2, 4, -2, 3]
arr = [-1, -2, -3, -4, -5]


def max_contiguous_subarray(arr, verbose=False):
    max_so_far = 0
    max_ending_here = 0

    for i in range(len(arr)):
        max_ending_here += arr[i]
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
        # IMPORTANT: CHECK IF max_ending_here < 0 (WHY? anytime our running max is negative, restart the sum
        # as we would never include a subset that sums to be negative overall
        if max_ending_here < 0:
            max_ending_here = 0

        if verbose:
            print "arr[i] = ", arr[i]
            print "max ending here = ", max_ending_here
            print "max_so_far = ", max_so_far
    return max_so_far


max_contiguous_subarray(arr)


# KADANES ALGORITHM (UTILIZING A RNNING MIN) - GREEN BOOK p181
# ========================================================================

def max_cont_subarray_2(arr):  # includes trough to peak indices
    T = arr[0]
    running_max = T
    running_min = min(0, T)
    a = 0;
    b = 0
    for j in range(1, len(arr)):
        T = T + arr[j]
        if T - running_min > running_max:
            running_max = T - running_min  # we found a new max
            b = j  # save the index (trough)
        if T < running_min:
            running_min = T  # we found a new min
            a = j  # save the index (peak)
    return running_max, a, b


arr = [1, 2, -5, -8, -2, 2, 6, 8, -1, -2, 5]
max_contiguous_subarray(arr)
max_cont_subarray_2(arr)

# ************************************************************************************
# TWO SUM (2sum) PROBLEM (LEET CODE)
# given an array of sorted numbers, find 2 numbers which sum up to a target number,
# return the indices
# ************************************************************************************

numbers = [1, 3, 4, 5, 5, 7, 9]
target = 9
twoSum(numbers, target)


def twoSum(numbers, target):
    # 2 pointers
    start = 0
    end = len(numbers) - 1

    while start < end:
        sum = numbers[start] + numbers[end]
        if sum > target:  # sum too big, decrement end
            end -= 1
        elif sum < target:  # sum too small, increment start
            start += 1
        elif sum == target:
            # return_string = "index1=",start,"index2=",end
            return start + 1, end + 1


# ************************************************************************************
# Longest substring without repeating chars - PROBLEM (LEET CODE)
# Given a string, find the length of the longest substring without repeating characters.
# e.g. Given "abcabcbb", the answer is "abc", which the length is 3.
# ************************************************************************************
s = 'dvdf'
lengthOfLongestSubstring(s)
s[0:1]


def lengthOfLongestSubstring(s):
    max_ending_here = 0
    max_so_far = 0
    max_index_ending_here = 0
    letters_seen = {}
    for i in range(0, len(s)):
        # if we've seen this letter before, reset the counter
        if letters_seen.has_key(s[i]) and s[i - 1] != s[i]:
            max_ending_here = 2
        elif letters_seen.has_key(s[i]) and s[i - 1] == s[i]:
            max_ending_here = 1
        else:  # new character
            max_ending_here += 1
            if max_ending_here > max_so_far:
                max_so_far = max_ending_here
                max_index_ending_here = i
            # add to hashmap to keep track of letters weve seen before
            letters_seen[s[i]] = 1
    # return max_so_far
    return max_so_far, s[(max_index_ending_here - max_so_far + 1):max_index_ending_here + 1]
    """
    :type s: str
    :rtype: int
    """


# ************************************************************************************
# EPI - Ch 17 Dynamic Programming - p273 - Example
# Find max sum over all subarrays
# SUPPORTS ALL CASES INCL ALL NEGATIVE NUMBERS
# ************************************************************************************
def max_sub_array_2(arr, verbose=False):
    min_sum = 0
    sum_here = 0
    max_sum = 0
    for i in range(len(arr)):
        sum_here += arr[i]
        if sum_here < min_sum:
            min_sum = sum_here  # new trough
        if sum_here - min_sum > max_sum:  # new peak
            max_sum = sum_here - min_sum
        if verbose:
            print "arr[i] = ", arr[i]
            print "sum ending here = ", sum_here
            print "max_sum_so_far = ", max_sum
            print "min_sum_so_far = ", min_sum
    return max_sum


max_sub_array_2(arr, verbose=True)

# ************************************************************************************
# EPI - Ch 25 Honors Class - 25.5 Compute longest increasing contiguous subarray
# longest contiguous increasing subarray, monotonic
# ************************************************************************************

# Longest subarray ending at j+1
#   1. A[j+1], if A[j+1] <= A[j]
#   2. longest subarray ending at j + A[j+1], if A[j+1] > A[j]
#   3. Keep additional variables to store max_so_far, index, length, etc


# Matrix Multiplication
# ************************************************************************************
A = [[1, 2], [3, 4], [5, 6]]
A[1][1]
len(A)  # 3
len(A[0])  # 2
B = [[1, 2, 3], [3, 2, 1]]
A = [1, 2, 3, 4, 5]
B = [[cols for cols in range(1, 2)] for rows in range(0, 5)]
# B = [[y for y in range(1,3)] for x in range(3,5)]
A.shape


def matrix_multiply(A, B):
    # compute dimensions
    # A = [n,m], B= [p,q]
    n = len(A)  # num rows
    m = len(A[0])  # num cols
    p = len(B)
    q = len(B[0])

    if m != p:
        raise NameError('Matrix Dimensions are not compatible for multiplication')  # ERROR, DIMENSIONS DONT MATCH
    C = [[0 for cols in range(q)] for rows in range(n)]  # new matrix will be n * q
    for row in range(n):  # row by row
        for col in range(q):  # for each col
            sum = 0
            for i in range(p):  # each element
                sum += A[row][i] * B[i][col]
            C[row][col] = sum
    return C


matrix_multiply(A, B)


# ************************************************************************************
# Ch 7 - EPI - Strings p7.1 p 86
# Write program that takes a string representing an integer and return the interger,
# and vice versa
# ************************************************************************************

def int_to_string(num):
    # hint: x mod 10, x/10
    while x:
        last_digit = x % 10
        remaining_digits = x / 10
        # STILL IN PROGRESS


        # ************************************************************************************


# Implement mergesort in python
# WORKS but needs review
# ************************************************************************************

# Merge 2 sorted arrays
m = mid - low
n = hi - (mid + 1)


def merge(left, right):  # merge 2 arrays
    c = []
    a, b = 0, 0
    # left = [left]
    # right = [right]
    while a < len(left) and b < len(right):  # 2 pointers - a,b
        if left[a] < right[b]:  # add min(a,b) to our new array, increment pointer
            c.append(left[a])
            a += 1
        elif left[a] > right[b]:
            c.append(right[b])
            b += 1
        else:  # left[a] == right[b]
            c.append(left[a])  # if equal, add both, increment both pointers
            c.append(right[b])
            a += 1
            b += 1
    # need to add leftovers
    while a < len(left):
        c.append(left[a])
        a += 1
    while b < len(right):
        c.append(right[b])
        b += 1
    return c


# ======== TEST ==========
left = [1, 3, 5]
right = [2, 6, 7]
merge(left, right)


# Merge Sort - working

def mergeSort(s):
    if len(s) < 2:
        return s
    else:
        mid = int(len(s) / 2)
        left = mergeSort(s[:mid])
        right = mergeSort(s[mid:])
        print "calling merge(left,right) on", left, right
        return merge(left, right)


s = [randint(1, 9) for x in range(20)]

mergeSort(s)


# ************************************************************************************
# Implement quicksort in python
# STILL IN PROGRESS
# ************************************************************************************

def quicksort(s, low, high):
    if ()


def partition(s, low, high):
    p = high  # set last item as partition pivot (this can be chosen to be anywhere)
    high = high - 1

    while (low < high):
        # increment low, decrement high, until both numbers are "out of place"
        # out of place - meaning, number in low > pivot and number in high < pivot
        # swap these two
        print s, low, high
        if s[low] >= s[p] and s[high] <= s[p]:  # NEED TO SWAP
            print "SWAPPING"
            swap(s, low, high)
            high -= 1
            low += 1
        elif s[high] >= s[p]:
            high -= 1
        elif s[low] <= s[p]:
            low += 1
        else:
            high -= 1
            low += 1
    # swap low with pivot
    swap(s, low, p)
    return s


a = [randint(1, 9) for x in range(10)]
# swap(a,1,2)
s = partition(a, 0, 9)


def quicksort(s, high, low):
    if low < high:
        p = partition()


# PYTHONIC WAY - found online
# ==========================

def quick_sort(l):
    if len(l) == 0:
        return l
    pivot = l[0]
    pivots = [x for x in l if x == pivot]
    smaller = quick_sort([x for x in l if x < pivot])
    larger = quick_sort([x for x in l if x > pivot])
    return smaller + pivots + larger


a = [randint(1, 100) for x in range(100)]
a
quick_sort(a)


# ************************************************************************************
# Implement linked list
# ************************************************************************************

class node:
    # constructor
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next = next_node

    def __str__(self):
        return str(self.data)


class linked_list:
    def __init__(self, head=None):
        self.head = head

    def append(self, data):
        # if its empty
        current = self.head
        if current is None:
            self.head = node(data)
        else:
            # traverse the list
            while current.next is not None:
                current = current.next
            current.next = node(data)

    def remove(self, data):
        current = self.head
        # if we are removing the head
        if current.data == data:
            self.head = current.next
        else:
            while current.next.data != data:
                current = current.next
            # stop if next is the one to delete
            current.next = current.next.next

    def __str__(self):
        # traverse the list
        if self.head == None:
            return "Empty List"
        else:
            total_string = ''
            current = self.head
            while current is not None:
                value_string = str(current.data)
                current = current.next
                total_string += ' -> ' + value_string
        return total_string


# ===========


l = linked_list()
l.append(1)
l.append(2)
l.append(3)
print l

l.remove(2)
print l

# ************************************************************************************
# Ch 8 - EPI - Linked Lists q8.1 on p 102
# Problem 8.1
# We have 2 singly linked lists, each node holds a number. Each list is sorted, Merge the
# lists so that they are still in order
# ************************************************************************************

l1 = linked_list()
l2 = linked_list()

l1.append(1)
l1.append(3)
l1.append(5)
l2.append(2)
l2.append(6)
l2.append(7)
l2.append(8)
print l1, l2

# initialize new linked list
l_new = linked_list()

# compare heads, take min
cur1 = l1.head
cur2 = l2.head

# set temp vars
temp1 = cur1.next
temp2 = cur2.next
if cur1.data < cur2.data:
    min_node = cur1
    other_node = cur2
else:
    min_node = cur2
    other_node = cur1

# compare other node vs next node
if other_node < min_node.next:
    # order should be min_node -> other_node -> min_node.next
    other_node.next = min_node.next
    min_node.next = other_node

l_new = linked_list(min_node)
print l_new

print min_node
print other_node.next


# use recursion

# NOT WORKING...
def merge_lists(node_a, node_b):
    # base case, node is null, weve reached the end
    if node_a == None:
        return node_b
    if node_b == None:
        return node_a

    if node_a.data < node_b.data:
        smaller_node = node_a
        smaller_node.next = merge_lists(node_a.next, node_b)
    else:
        smaller_node = node_b
        smaller_node.next = merge_lists(node_a, node_b.next)
    return smaller_node


new_l = merge_lists(l1.head, l1.head)

print new_l


def merge_lists(left_head, right_head):
    c = []
    left_ptr = left_head
    right_ptr = right_head
    while left_ptr and right_ptr:
        if left_ptr.data < right_ptr.data:
            c.append(left_ptr.data)
            left_ptr = left_ptr.next
        elif right_ptr.data < left_ptr.data:
            c.append(right_ptr.data)
            right_ptr = right_ptr.next
        else:  # left[a] == right[b]
            c.append(left_ptr)
            c.append(right_ptr)
            left_ptr = left_ptr.next
            right_ptr = right_ptr.next
    # leftovers
    while left_ptr:
        c.append(left_ptr.data)
        left_ptr = left_ptr.next
    while right_ptr:
        c.append(right_ptr.data)
        right_ptr = right_ptr.next
    return c


merge_lists(l1.head, l2.head)


# found online - recursive merge lists

def MergeLists(headA, headB):
    # base cases
    if headA is None and headB is None:
        return None
    if headA is None:
        return headB
    if headB is None:
        return headA
        # recursion
    if headA.data < headB.data:
        new_node = headA
        new_node.next = MergeLists(headA.next, headB)
    else:
        new_node = headB
        new_node.next = MergeLists(headA, headB.next)
    return new_node


print l1, l2
new_head = MergeLists(l1.head, l2.head)
l3 = linked_list(new_head)
print l3

current = new_head
while current is not None:
    print current.data
    current = current.next

# REVERSE A LINKED LIST
# ==============================
print l1
print reverse_list_iter(l1)


def reverse_list_iter(l):
    if l.head is None:
        return None
    elif l.head.next is None:
        return l
    else:
        prev = l.head
        current = l.head.next
        while current is not None:
            print current.data
            # first save the next node
            nextNode = current.next
            current.next = prev
            prev = current
            current = nextNode
    l.head = prev
    return l


def reverse_list(l):


# helper function for recursion
def reverse_list_helper(current):
    # base case
    if current.next is None:
        return current
    else:
        last = reverse_list_helper(current.next)
        last.next = current


# Example on p117
# print linked list in reverse using recursion

def print_list_reverse(l):
    traverse_list(l.head)


def traverse_list(current):
    if current.next is None:
        print current
    else:
        traverse_list(current.next)
        print current


print l1
print_list_reverse(l1)


# print linked list in reverse using a stack

def print_list_reverse_stack(l):
    stck = []
    current = l.head
    while current != None:
        stck.append(current.data)
        current = current.next
    while stck:
        print stck.pop()


print_list_reverse_stack(l1)


# ************************************************************************************
# 8.7 - EPI p 109 - Remove the k-th last element from a list
# ************************************************************************************
# Here, the trick is to iterate 2 ptrs, one at head, the other at head + k, when
# faster pointer hits end, we know the slower one is where we want
def remove_kth_last_element(l, k):
    slow_ptr = l.head
    fast_ptr = l.head
    for i in range(k):  # advance fast ptr k nodes
        fast_ptr = fast_ptr.next
        if fast_ptr is None:  # k is > size of list
            return 'Nil'

    # advance both ptrs together
    while fast_ptr.next is not None:
        fast_ptr = fast_ptr.next
        slow_ptr = slow_ptr.next
    # now that fast ptr is at the end, slow_ptr must point to the k+1 last node
    slow_ptr.next = slow_ptr.next.next


# TEST
l = linked_list()
for i in range(1, 11):
    l.append(i)
print l

remove_kth_last_element(l, 20)
print l

# ************************************************************************************
# Recursion
# ************************************************************************************

# GOOD RECURSION EXAMPLE from Brett Bernstein
n = 3


def p(n):
    if n == 0:
        return
    print "before: ", n
    p(n - 1)
    print "after: ", n


p(3)


# Implement binary search
# make sure to include "return" on "return binary search(...)
def binary_search(arr, left, right, target):
    # base case
    if left == right:
        return -1  # not found
    mid = (left + right) / 2
    if arr[mid] == target:  # found it
        return mid
    elif target < arr[mid]:  # go lower
        return binary_search(arr, left, mid, target)
    else:
        return binary_search(arr, mid + 1, right, target)


arr = [randint(1, 9) for x in range(10)]
arr
print binary_search(arr, 0, len(arr) - 1, 4)
found
(len(arr) - 1) / 2


# ************************************************************************************
# ch 9 - Stacks and Queues - EPI - p117
# ************************************************************************************
# Implement a stack, stack = LIFO, similar to linked list

class node:
    # constructor
    def __init__(self, data=None, next_node=None):
        self.data = data
        self.next = next_node

    def __str__(self):
        return str(self.data)


class stack:
    def __init__(self, head=None):
        self.head = head

    def push(self, data):
        # if its empty
        current = self.head
        if current is None:
            self.head = node(data)
        else:
            # add to begining
            new_head = node(data)
            new_head.next = self.head
            self.head = new_head

    def pop(self):
        if self.head is None:
            return None
        to_remove = self.head
        self.head = self.head.next
        return to_remove.data

    def peek(self):
        if self.head is None:
            return None
        return self.head.data

    def __str__(self):
        # traverse the list
        if self.head == None:
            return "Empty List"
        else:
            total_string = ''
            current = self.head
            while current is not None:
                value_string = str(current.data)
                current = current.next
                total_string += ' -> ' + value_string
        return total_string


# ===================

s = stack()
s.push(1)
s.push(2)
s.push(3)
s.peek()
s.pop()
print s
s.pop()
s.pop()


# ************************************************************************************
# 9.1 Implement a stack with Max (and Min) API
# ************************************************************************************
# NEED TO FIX: NEED TO ADD running min, not just min

class stack_max_min:
    def __init__(self, head=None):
        self.head = head
        self.max = stack()
        self.min = stack()

    def push(self, data):
        # if its empty
        current = self.head
        if current is None:
            self.head = node(data)
            # update max and min
            self.max.push(data)
            self.min.push(data)
        else:
            # add to begining
            new_head = node(data)
            new_head.next = self.head
            self.head = new_head
            # update max or min
            if data >= self.max.peek():  # we have a new max
                self.max.push(data)
            if data <= self.min.peek():  # we have a new min
                self.min.push(data)

    def pop(self):
        if self.head is None:
            return None
        to_remove = self.head
        self.head = self.head.next
        # update max or min
        if to_remove.data == self.min.peek():  # we are removing current min
            self.min.pop()
        if to_remove.data == self.max.peek():  # we are removing current max
            self.max.pop()
        return to_remove.data

    def peek(self):
        return self.head.data

    def get_max(self):
        return self.max.peek()

    def get_min(self):
        return self.min.peek()

    def __str__(self):
        # traverse the list
        if self.head == None:
            return "Empty List"
        else:
            total_string = ''
            current = self.head
            while current is not None:
                value_string = str(current.data)
                current = current.next
                total_string += ' -> ' + value_string
        return total_string


# ===========

s = stack_max_min()
for i in [2, 3, 2, 5, 1, 6, 9, 1, 2]:
    s.push(i)
    print s, " : ", s.get_max(), s.get_min()
for i in range(9):
    s.pop()
    print s, " : ", s.get_max(), s.get_min()


# ************************************************************************************
# 9.9 Implement a Stack using 2 Queues
# ************************************************************************************

class MyQueue(object):
    def __init__(self):
        self.first = []
        self.second = []

    # look at top element
    def peek(self):
        if len(self.second) == 0:  # Only do this if 2nd stack is empty
            while len(self.first) > 0:
                self.second.append(self.first.pop())
        if len(self.second) == 0:  # if its still empy
            return 'empty'
        else:
            return self.second[len(self.second) - 1]

    def pop(self):
        if len(self.second) == 0:  # Only do this if 2nd stack is empty
            while len(self.first) > 0:
                self.second.append(self.first.pop())
        if len(self.second) == 0:  # if its still empy
            return 'empty'
        else:
            return self.second.pop()

    def put(self, value):
        # put in stack 1
        self.first.append(value)


q = MyQueue()
q.put(1)
q.put(2)
q.put(3)

q.pop()
q.peek()
q.pop()

# ************************************************************************************
# Ch 11 - Heaps EPI p158
# ************************************************************************************
import heapq as hq

# 11.5 Compute running median
# ************************************************************************************
# TRICK: maintain 2 heaps, a min and max heap

min_heap = []
max_heap = []
hq.
for i in [randint(0, 99) for x in range(20)]:
    heapq.heappush(min_heap, i)
print min_heap
?heapq

len(min_heap)


# min and max heaps [max heap] [median] [min heap]
def running_median(n, min_heap, max_heap):
    if len(min_heap) == 0 and len(max_heap) == 0:
        hq.heappush(min_heap, n)
        return min_heap[0]
    else:
        if n >= min_heap[0]:
            # put in the min_heap
            hq.heappush(min_heap, n)
        else:  # put in max heap
            hq.heappush(max_heap, -n)
        if len(min_heap) > len(max_heap) + 1:
            # uneven, pop from min_heap to max heap
            hq.heappush(max_heap, -hq.heappop(min_heap))
        elif len(max_heap) > len(min_heap):
            hq.heappush(min_heap, -hq.heappop(max_heap))
        if len(max_heap) == len(min_heap):  # if equal lengths, return average
            return 0.5 * (min_heap[0] - max_heap[0])
        else:
            return min_heap[0]


for i in [10, 1, 2, 3, 4, 5]:
    print running_median(i, min_heap, max_heap)


# HACKER RANK (DOESNT PASS ALL TESTS)
def calculate_median(n, min_heap, max_heap):
    # if heaps are empty
    if len(min_heap) == 0 and len(max_heap) == 0:
        hq.heappush(min_heap, n)
        return float(min_heap[0])
    else:
        # if a < min_heap[0] => put in max_heap, else put in min_heap
        if n >= min_heap[0]:
            hq.heappush(min_heap, n)
        else:
            hq.heappush(max_heap, -n)  # negate, as all heaps are min_heaps
        # rebalance heaps if they get too out of balance
        if len(min_heap) > len(max_heap) + 1:
            hq.heappush(max_heap, -hq.heappop(min_heap))  # negate before pushing into max heap
        # if odd numbers, just return min_heap[0], else average the two
        if (len(min_heap) + len(max_heap)) % 2 == 0:
            return 0.5 * (min_heap[0] - max_heap[0])  # dont forget to negate the max_heap
        else:
            return float(min_heap[0])


# TICKET SALES
# ************************************************************************************
"""
1. Cre­ate a max-heap of size of num­ber of win­dows. (Click here read about max-heap and pri­or­ity queue.)
2. Insert the num­ber of tick­ets at each win­dow in the heap.
3. Extract the ele­ment from the heap k times (num­ber of tick­ets to be sold).
4. Add these extracted ele­ments to the rev­enue. It will gen­er­ate the max rev­enue since extract­ing for heap will give you the max ele­ment which 
is the max­i­mum num­ber of tick­ets at a win­dow among all other win­dows, and price of a ticket will be num­ber of tick­ets remain­ing at each 
window.
5. Each time we extract an ele­ment from heap and add it to the rev­enue, reduce the ele­ment by 1 and insert it again to the heap since after 
num­ber of tick­ets will be one less after selling.
"""


def maximumAmount(a, k):
    pq = queue.PriorityQueue()  # well use a min-heap and just negate all the values

    for i in a:
        pq.put(-i)  # negate values
    total = 0
    for i in range(k):
        sales = pq.get()  # sell the maximum which is at root of heap
        total += sales
        pq.put(sales + 1)  # put back maximum -1 into the heap

    return -total


# ************************************************************************************
# Ch 12 - Searching 0 EPI p172
# ************************************************************************************

# Search a sorted array for 1st occurance of K

def search_array_first(arr, k):
    left = 0
    right = len(arr) - 1
    while left < right:
        mid = (left + right) / 2
        if k < arr[mid]:
            right = mid - 1
        elif k == arr[mid]:  # found one occurence
            right = mid
        else:  # k > arr[mid]
            left = mid + 1
    return right


arr = [1, 3, 4, 4, 5, 5, 5, 5, 5, 5, 7, 8, 9, 9]
k = 5

search_array_first(arr, k)


# ************************************************************************************
# EPI - 12.8 0 Find the k-th largest element (similar to find the median of array), p 180
# kth largest element
# ************************************************************************************

# BRUTE FORCE: sort -> O(nlogn)

# BETTER:
# USE MIN HEAP, store the top k largest elements, if new num < k, discard. O(nlogk)

# BEST:
# TRICK: pivot, similar to quicksort -> O(n) on average
# pick random "pivot", bucket all < pivot to the left, all > pivot to the right
# each time, discarding (half) the array

# IN PROGRESS

# ************************************************************************************
# EPI - Ch 25 Honors Class - 25.15 - Search array of unknown length
# ************************************************************************************
# USe binary search
# increment by 2^i
# as soon as we find invalid value, say 2^i - 1, use binary search on interval [2^(i-1), 2^i-2]
# O(logn)


# ************************************************************************************
# Ch 13 - Hash Tables EPI p190
# ************************************************************************************
# 13.1
# Write a function to test whether the letters forming a string can be permuted to form a palindrome
# (spells the same backwards as forwards)
# form palindrome if: num letters is even, each has 2 occurences
#                      num letters is odd, each has 2 occurences, one has one occurence
def can_form_palindrome(s):
    my_dict = {}

    for i, letter in enumerate(s):
        if my_dict.has_key(letter):
            my_dict[letter] += 1
        else:
            my_dict[letter] = 1

    counts = []
    # once dict is populated, count all occurences
    for i, j in my_dict.iteritems():
        counts.append(j)

    mods = sum(map(lambda x: int(x) % 2, counts))  # sum up each reminder

    if len(s) % 2 == 0:  # IF EVEN
        return mods == 0
    else:
        return mods == 1


# TESTING
s = 'levels'
print can_form_palindrome(s)


# ************************************************************************************
# Ch 14 - Sorting EPI p213
# ************************************************************************************
# 14.1
# Intersection of 2 sorted arrays, intersecting sorted arrays
# Write a function that takes as inputs, 2 sorted arrays, returns new array containing
# elements present in BOTH input arrays (no dupes)

# ALGO
# - 2 pointers 1 to a, 1 to b
# - increment each and compare
# - if both equal, add to new array, increment both
# - Complexity = O(n + m)
def intersect_arrays(arr_a, arr_b):
    ptr_a = 0
    ptr_b = 0
    out = []

    while ptr_b <= len(arr_b) - 1 and ptr_a <= len(arr_a) - 1:
        if arr_a[ptr_a] == arr_b[ptr_b]:  # we have a match
            if len(out) == 0:
                last_value = None
            else:
                last_value = out[len(out) - 1]  # save last value
            if last_value != arr_a[ptr_a]:  # make sure u dont add dupes
                out.append(arr_a[ptr_a])
            ptr_b += 1
            ptr_a += 1
        elif arr_a[ptr_a] > arr_b[ptr_b]:
            ptr_b += 1  # increment ptr_B
        elif arr_b[ptr_b] > arr_a[ptr_a]:
            ptr_a += 1  # increment ptr a
    return out


arr_a = [2, 3, 3, 4, 4, 5, 6, 6, 8, 10, 12]
arr_b = [5, 5, 6, 8, 8, 9, 10, 10]

print intersect_arrays(arr_a, arr_b)
out = []
len(out)


# 14.2
# ================================
# Merge 2 sorted arrays
# Write a function that takes as inputs, 2 sorted arrays, merged arrays in place of
# 1st array
# O(m+n)

# TRICK -> fill in starting from the end (to avoid shifting)
# m = length of arr_a, n = length of arr_b
def merge_2_sorted_arrays(arr_a, arr_b, m, n):
    a = m - 1
    b = n - 1
    c = m + n - 1  # start at end
    while a >= 0 and b >= 0:
        if arr_a[a] > arr_b[b]:
            arr_a[c] = arr_a[a]
            a -= 1
        elif arr_b[b] >= arr_a[a]:
            arr_a[c] = arr_b[b]
            b -= 1
        c -= 1
        # print arr_a, a, b,c

    # we still may have some left over in B
    while b >= 0:
        arr_a[c] = arr_b[b]
        b -= 1
        c -= 1
    return arr_a


arr_a = [2, 3, 3, 4, '', '', '', '', '', '']
arr_b = [1, 1, 1, 5, 6]
m = 4
n = 5
print merge_2_sorted_arrays(arr_a, arr_b, m, n)


# ************************************************************************************
# EPI - problem 14.5 - Merging Intervals, p218
# Take as input an array of disjoint closed intervals and an interval to be added and returns union
# of all intervals
# ************************************************************************************
# Algo:
# Iterate through all intervals that appear completely before the interval to be added (call it A)
# once we encounter an interval which intersects A, compute its union and add it
# compare newly formed interval to the next interval, union if necessary
# iterate through remaining intervals
# Complexity -> O(n)

# ************************************************************************************
# Ch 15 - Binary Search Trees - EPI - p 253
# ************************************************************************************

class bst_node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def insert(self, data):
        if data < self.data:
            if self.left is None:
                self.left = bst_node(data)
            else:
                self.left.insert(data)
        else:  # data > self.data
            if self.right is None:
                self.right = bst_node(data)
            else:
                self.right.insert(data)

    def print_in_order(self):
        # print in order traversal
        if self.left is not None:
            self.left.print_in_order()

        print self.data, " -> ",

        if self.right is not None:
            self.right.print_in_order()


root = bst_node(50)
for i in [randint(1, 99) for x in range(5)]:
    root.insert(i)

root.print_in_order()

# 15.1 Test if BST satisfies the BST condition
# ==================================================
# TRICK: when traversing the tree, check to make sure each nodes value falls within an interval [-infinity, root] (for left subtree)
# or [root, parent root] and keep updating the min,max intervals

int_min = -999999
int_max = 999999


def BST_check(root, my_min, my_max):
    if root is None:
        return
    elif root.data > my_max or root.data < my_min:  # if this node is outside our range
        return "False"
    else:
        my_data = root.data
        print "checking: ", my_data, " in (", my_min, " : ", my_max, ")"
        BST_check(root.left, my_min, my_data)
        BST_check(root.right, my_data, my_max)
        return "True"


BST_check(root, int_min, int_max)


# ************************************************************************************
# Ch 16 - Recursion - EPI - p 253
# ************************************************************************************

# Example - calculate GCD of 2 numbers
# ************************************************************************************
# hint -> gcd(x,y) = gcd(x, y-x) if y > x
# this implied gcd(x,y) = gcd(x, y mod x)

def gcd(x, y):
    if y == 0:
        return x
    elif y > x:
        return gcd(x, y % x)
    elif y < x:
        return gcd(y, x % y)
    else:  # x == y
        return x


gcd(144, 8)

# ************************************************************************************
# 16.1 - Towers of Hanoi
# ************************************************************************************
# represent 3 towers as 3 individual stacks
s1 = [3, 2, 1]
s2 = []
s3 = []


def towers_hanoi(n  # how many remaining
                 , source  # source pile (a stack)
                 , helper  # helper pile ( a stack)
                 , target):  # target pile, a stack):
    if n > 0:
        print s1, s2, s3
        # move n-1 tower over to helper
        towers_hanoi(n - 1, source, target, helper)

        if len(source) > 0:  # if source not empty
            # move from source to target
            target.append(source.pop())
            print source, target
        # move from helper to target
        towers_hanoi(n - 1, helper, source, target)


towers_hanoi(3, s1, s2, s3)
print s1, s2, s3


# DONT USE RETURN STATEMENT HERE!!

# ************************************************************************************
# Ch 17 - Dynamic Programming - EPI - p 272
# ************************************************************************************

# Example - fibonacci
# ************************************************************************************

def fib_slow(n):
    if n < 2:
        return n
    else:
        return fib_slow(n - 1) + fib_slow(n - 2)


fib_slow(50)

fib(50)

# Enhanced version (cache values in a dict?)
cache = {}


def fib(n):
    if n < 2:
        return n
    else:
        # check if f(n-1) and f(n-2) exist in our dict
        # if not, calculate it (use recursion)
        if cache.has_key(n - 1):
            fib_n_1 = cache[n - 1]
        else:
            fib_n_1 = fib(n - 1)
            cache[n - 1] = fib_n_1
        if cache.has_key(n - 2):
            fib_n_2 = cache[n - 2]
        else:
            fib_n_2 = fib(n - 2)
            cache[n - 2] = fib_n_2
        return fib_n_1 + fib_n_2


# BETTER, FOUND ONLINE
# https://jeremykun.com/2012/01/12/a-spoonful-of-python/
def fib(n):
    saved_fib = [0, 1]
    for i in range(2, n + 1):
        saved_fib.append(saved_fib[i - 1] + saved_fib[i - 2])
    return saved_fib[n]


for i in range(10):
    print fib(i)

# ************************************************************************************
# HACKER RANK - compute factorial
# ************************************************************************************

# save calculated factorials
saved_facts = {}


def factorial(n):
    if n <= 2:
        return n
    else:
        if n not in saved_facts:
            saved_facts[n] = n * factorial(n - 1)
    return saved_facts[n]


factorial(15)


# ************************************************************************************
# 17.1 - Football Scores - write program that takes final score and outputs number of combinations
# of plays that result in the final score
# NOTE: same as the coin change problem
# ************************************************************************************

# NOT WORKING
def football(n):
    # define base cases
    if n <= 0:
        return 0
    elif n == 3 or n == 2:
        return 1
    else:
        return 1 + football(n - 2) + football(n - 3) + football(n - 7)


football(12)

n = 12
scores = [2, 3, 7]


# wrap into a function

def football(n, scores):  # Similar to knapsack problem
    # Need to populate the table
    # rows = number of point combos (2,3,7)
    # cols = total score, up to 12
    A = np.zeros([len(scores), n + 1])

    # 1st row
    for col in range(0, n + 1, 2):
        A[0, col] = 1
    for row in range(1, len(scores)):
        for col in range(0, n + 1):  # e.g. start at 2
            if col < scores[row]:
                A[row, col] = A[row - 1, col]  # less than the current number, copy over from above row
            else:
                A[row, col] = A[row - 1, col] + A[row, col - scores[row]]

    return A[len(scores) - 1, n]


football(12, [1, 2, 3])


# WHAT IF NUMPY IS NOT ALLOWED?
# THIS WORKS
# SAME AS COIN CHANGE PROBLEM

def football(n, scores):
    # Need to populate the table
    # rows = number of point combos (2,3,7)
    # cols = total score, up to 12
    # A = np.zeros([len(scores),n+1])

    # just in case u cant use numpy
    A = [[0 for col in range(n + 1)] for row in range(len(scores))]

    # 1st row
    for col in range(0, n + 1, scores[0]):
        A[0][col] = 1
    # for the rest of the rows
    for row in range(1, len(scores)):
        for col in range(0, n + 1):  # e.g. start at 2
            if col < scores[row]:
                A[row][col] = A[row - 1][col]  # less than the current number, copy over from above row
            else:
                A[row][col] = A[row - 1][col] + A[row][col - scores[row]]

    return A[len(scores) - 1][n], A


answer, A = football(4, [1, 2, 3])
A


# ************************************************************************************
# 17.3 - Count number of ways to traverse a 2D array
# ************************************************************************************
# works! but needs review
# Recursion plus cache -> keep track of visited nodes with num_ways matrix
def count_ways_to_xy(x, y, num_ways):
    # base case
    if x == 0 and y == 0:  # we are at the origin
        return 1
    if num_ways[x, y] == 0:  # not set yet
        if x == 0:
            ways_x = 0
        else:
            ways_x = count_ways_to_xy(x - 1, y, num_ways)
        if y == 0:
            ways_y = 0
        else:
            ways_y = count_ways_to_xy(x, y - 1, num_ways)
        num_ways[x, y] = ways_x + ways_y
    print x, y, num_ways[x, y]
    return num_ways[x, y]


n = 5
num_ways = np.zeros([n, n])
count_ways_to_xy(n - 1, n - 1, num_ways)
num_ways


# =========== final submission for Hackerrank (Blue Mountain Capital Coding Test) ======================
# WORKING!!!

# Complete the function below.
def canReach(a, b, c, d):
    num_ways = [[0 for y in range(d + 1)] for x in range(c + 1)]
    if a > c or b > d:  # if starting position is greater than ending position, this is impossible
        return 'No'
    if count_ways_to_xy(c, d, a, b, num_ways) > 0:
        return 'Yes'
    else:
        return 'No'


def count_ways_to_xy(x, y, a, b, num_ways):
    # if we found a way, break
    if num_ways[a][b] > 0:
        return 1
    # base case
    if x == a and y == b:  # we are at the origin
        return 1
    if x < a or y < b or x == 0 or y == 0:
        return 0

    if num_ways[x][y] == 0:  # not set yet
        if x < a or x - y < 0:  # passed origin, no paths found
            ways_x = 0
        else:
            ways_x = count_ways_to_xy(x - y, y, a, b, num_ways)
        if y < b or y - x < 0:  # passed origin, no paths found
            ways_y = 0
        else:
            ways_y = count_ways_to_xy(x, y - x, a, b, num_ways)
        num_ways[x][y] = ways_x + ways_y
    return num_ways[x][y]


# ************************************************************************************
# 17.6 - The Knapsack Problem - works but needs review
# ************************************************************************************

# our saved max profit matrix
values = [60, 50, 70, 30]
weights = [5, 3, 4, 2]
capacity = 5
V = np.zeros([len(values), capacity + 1])


def knapsack(values, weights, capacity):
    # if our total capacity is 0, then our optimal value has to be 0
    V[:, 0] = 0
    for i in range(len(values)):
        for w in range(capacity + 1):  # need to start at 0
            # if current object weights too much
            if weights[i] > w:
                V[i, w] = V[i - 1, w]  # cant add the new item
            else:  # take max of 1) take this item, 2) dont take this item
                V[i, w] = max(V[i - 1, w - weights[i]] + values[i], V[i - 1, w])
    return V[i, w]


knapsack(values, weights, capacity)
print V

# ************************************************************************************
# 17.9 - Pick up coins for max gain - p291
# ************************************************************************************

coins = [25, 5, 10, 5, 10, 5, 10, 25, 1, 25, 1, 25, 1, 25, 5, 10]

coins = [10, 15, 30, 20]


# save all profits in a table of (a,b)
# profit = np.zeros([len(coins), len(coins))
# keep pointers a,b
def coins_max_gain(a, b):
    # base case
    if a > len(coins) - 1 or b < 0:
        return 0
    if a >= b:
        return 0
    else:
        # pick a (plus min of next choice, as player B will minimize our choice)
        pick_a = coins[a] + min(coins_max_gain(a + 1, b - 1), coins_max_gain(a + 2, b))
        # pick b (plus min of next choice, as player B will minimize our choice)
        pick_b = coins[b] + min(coins_max_gain(a + 1, b - 1), coins_max_gain(a, b - 2))
        # profit[a,b] =  max(pick_a, pick_b)
        return max(pick_a, pick_b)
        # return profit[a,b]


coins_max_gain(0, len(coins) - 1)

# Recursive Staircase problem (Davis Staircase)
# ************************************************************************************

saved_staircase = {1: 1, 2: 2, 3: 4}


def staircase(n):
    # base case
    if n <= 3:
        return saved_staircase[n]
    elif n not in saved_staircase:
        saved_staircase[n] = staircase(n - 1) + staircase(n - 2) + staircase(n - 3)
    return saved_staircase[n]


n = 7
staircase(7)

# save into dict

# Hackerrank - Cracking the coding interview
# DP - Coin Change Problem
# Given a number of dollars, , and a list of dollar values for  distinct coins,
# find and print the number of different ways you can make change for  dollars if each coin
# is available in an infinite quantity.
# ************************************************************************************

# similar to generalized staircase problem
coins = [1, 2, 3]
saved_staircase = {}


def staircase(n):
    # base case
    if n <= 1:
        return 1
    elif n not in saved_staircase:
        for i in range(1, n)
            saved_staircase[n] = staircase(n - 1) + staircase(n - 2) + staircase(n - 3)
    return saved_staircase[n]


# ************************************************************************************
# Ch 18 - Greedy Algorithms and Invariants - EPI - p 272
# ************************************************************************************

# Example - coin change - make change for a given number of cents
# ************************************************************************************
coins = [1, 5, 10, 25, 100]


def make_change(n, coins):
    change = {}  # a dictionary to keep track of how many quarters, dimes, etc
    for i in reversed(coins):  # iterate in descending order
        if i <= n:
            num_this_coin = int(n / i)
            n = n % i
            change[i] = num_this_coin
    return change


make_change(194, coins)

# ************************************************************************************
# GS Interview Q - find the median of two sorted arrays
# ************************************************************************************


# ************************************************************************************
# Cracking Coding Interview - Sorting and Search - Ch 11
# 11.6 - 2-D Array Search, or sorted matrix search (M by N)
# OR EPI problem 12.6 p 177
# ************************************************************************************
# Algo:
# Start from the left-most column
# if num > target: move left
# if num < target: move down
# Complexity -> worst case, insect m+n-1 elements, so O(m+n)


# ************************************************************************************
# Leet Code - Is integer a perfect square? (hint, use binary search)
# ************************************************************************************

print isPerfectSquare(82)


# inefficient o(n) soln
def isPerfectSquare(num):
    for i in range(1, num / 2 + 1):
        sq = i * i
        if num == sq:
            return True
        elif num < sq:
            return False
    return False


def isPerfectSquare(num):
    n = num / 2 + 1
    while n <= num / 2 + 1:
        sq = i * i
        if num == sq:
            return True
        elif num < sq:
            return False
    return False
