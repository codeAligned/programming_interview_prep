# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Cracking the Coding Interview
# Created 7-6-2017
# Author: Alex H Chao
# ************************************************************************************
import math
import numpy as np
import pandas as pd
from helper.helper import *
from random import randint



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
                total_string += value_string + ' -> '
        return total_string

# ************************************************************************************
# Reverse a Linked List
# ************************************************************************************

def reverse_linked_list(l):
    reverse_linked_list_helper(l.head, None)

def reverse_linked_list_helper(cur, prev):
    if cur is None:
        return None
    next = cur.next # save the next node
    cur.next = prev # set next to prev (reversing)
    print('cur = {}, prev = {}'.format(next, prev))
    reverse_linked_list_helper(next, cur)

#iterative
def reverse_linked_list(head):
    # no base cases needed!
    prev = None
    while head:
        #first save next
        next = head.next # save next
        head.next = prev # reverse the next ptr
        head = head.next # move head over
        prev = head
    return prev

# ===========
#found online
class Solution:
# @param {ListNode} head
# @return {ListNode}
def reverseList(self, head):
    prev = None
    while head:
        curr = head
        head = head.next
        curr.next = prev
        prev = curr
    return prev
Recursion

class Solution:
# @param {ListNode} head
# @return {ListNode}
def reverseList(self, head):
    return self._reverse(head)

def _reverse(self, node, prev=None):
    if not node:
        return prev
    n = node.next
    node.next = prev
    return self._reverse(n, node)



l = linked_list()
l.append(1)
l.append(2)
l.append(3)
print l

n = node()
n = reverse_linked_list(l.head)
print(n)

list(l)
print(l)

l.remove(2)
print l


# ************************************************************************************
# Check if list has cycles
# ************************************************************************************




# ************************************************************************************
# 2.1 Remove duplicates from an unsorted linked list
# ************************************************************************************
# Methodology:
# Method 1) Keep track of nodes in a hash table / dict, o(n)
#           Keep track of prev, if we encounter a dupe, set previous.next = n.next
# Method 2) Use 2 pointers (runner), current iterates through list, runner checks all
# subsequent nodes for dupes




# ************************************************************************************
# 2.2 Find kth to last element of a singly linked list
# ************************************************************************************
# Methodology:
#  Have 2 pointers, increment the 1st pointer k times
#  Then, keep incrementing both pointers until ptr 2 reaches the end (ptr2 is the one
#  you want



# ************************************************************************************
# 2.3 Delete a note in middle of a singly linked list with only access to that node
# ************************************************************************************
# Methodology:
#  Copy data from the next node over to the current node, delete next node



# ************************************************************************************
# 2.4 Partition (pivot) a linked list around value x so that nodes < x before,
# nodes > x come after
# ************************************************************************************
# Methodology:
#  1) make 2 new lists and merge them
#  2) start one new double ended linked list (add to head or tail )



# ************************************************************************************
# 2.6 Given list with cycle, return the node at begining of the "loop"
# ************************************************************************************
# Methodology:
#  1) runner technique, 1 ptr moves 1 at a time, 2nd ptr moves 2 at a time
#  2) when they collide, move slowptr to the head. keep fast ptr as is
# 3) move each 1 at a time. when they meet again is the start of the loop

# taken from EPI
def has_cycle(head):
    fast = slow = head
    while fast and fast.next and fast.next.next:
        slow = slow.next
        fast = fast.next.next # in increment slow and fast
        if slow is fast:
            slow = head
            # increment both ptrs at same time
            while slow is not fast:
                slow = slow.next
                fast = fast.next
            return slow # this is when they meet again, slow is at start of cycle
    return None # no cycle



# ************************************************************************************
# 2.7 Check if linked list is a palindrome
# ************************************************************************************
# Methodology:
#  1) Reverse the 1st half of the list while inserting into a stack
#  2) Use the runner technique to find the middle (when fast ptr reaches end, slow ptr
#  is in the middle
#  3) increment slow ptr one by one and pop elements off stack


