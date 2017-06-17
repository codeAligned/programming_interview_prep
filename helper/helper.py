"""
Helper functions for programming interview questions

May 3 2017

Alex Chao

"""
import random


def convert_dict_to_dataframe(d):
    return pd.DataFrame.from_dict(d,orient='index')


# swap, pass by reference


def swap(a, i, j):
    temp = a[i]
    a[i] = a[j]
    a[j] = temp
    #return a



def random_matrix(n, m=None,min=0,max=9):
    if m is None:
        m = n
    rows = n
    columns = m
    return [[random.randrange(min, max+1) for x in range(columns)] for y in range(rows)]


def initialize_matrix(n):
    """
    intialize matrix as [1,2,3,4...]
    """
    return [[i+j*(n) for i in range(n)] for j in range(n)]



def initialize_matrix_zeros(rows,cols):
    return [[0 for x in range(cols)] for y in range(rows)]


"""
def swap(a, b):  # assume a > b
    a = a - b  # a = diff
    b = a + b  # b = diff + b = a
    a = b - a  # a = a- diff = b
    print(a, b)
"""
