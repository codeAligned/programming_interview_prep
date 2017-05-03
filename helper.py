"""
Helper functions for programming interview questions

May 3 2017

Alex Chao

"""


def initialize_matrix(n):
    return [[i+j*(n) for i in range(n)] for j in range(n)]

def swap(a, b):  # assume a > b
    a = a - b  # a = diff
    b = a + b  # b = diff + b = a
    a = b - a  # a = a- diff = b
    print a, b
