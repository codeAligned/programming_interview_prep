# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 7-13-2017
# Author: Alex H Chao
# ************************************************************************************

# Linked Lists, EPI, p 82


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
# 7.1 Merge 2 sorted linked lists
# ************************************************************************************
# solutions

def merge_two_sorted_lists(l1, l2):
    dummy_head = tail = node()
    while l1 and l2:
        if l1.data < l2.data:
            tail.next = l1
            next = l1.next
        else:
            tail.next = l2
            next = l2.next

    tail.next = l1 or l2
    return dummy_head.next



# ************************************************************************************
# 7.9 Cyclic Right Shift for Singly Linked Lists
# ************************************************************************************
# Methodology
#  - note: k may be larger than n, so need to take mod
#  - find tail node t
#  - set t.next = head
#  - new head is (n-k)th node


# ************************************************************************************
# 7.8 Remove Dupes from a Sorted List
# ************************************************************************************
# Methodology
# book solution
def remove_dupes(l):
    this = l
    while this:
        next_distinct = this.next

        while next_distinct and next_distinct.data == this.data: # dupe found
            next_distinct = next_distinct.next
        this.next = next_distinct
        this = next_distinct
    return l



