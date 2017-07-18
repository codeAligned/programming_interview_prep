import heapq
import math as m
import numpy as np
from helper.helper import *
from random import randint

# Traveling is Fun
# ===================
# There are n cities numbered from 1 to n
# 2 cities, x and y, are connected by a bidirectional road iff gcd(x,y) > some constant g
# return an array of q integers where the value of each index is 1 if a path exists from:
#   - originCities to destinationCities
# 0 otherwise

# ============
# represent graph as a matrix (n by n)
# 1 if gcd(x,y) > g, else 0
# then, translate matrix into an actual graph
# to find if a path exists, use DFS (or BFS)



def  connectedCities(n, g, originCities, destinationCities):

