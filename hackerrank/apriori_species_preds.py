# A Priori Coding Test
# 6-25-2017
# Hacker Rank
# ============================

import heapq
import math as m
import numpy as np
from helper.helper import *
from random import randint


# minimum number of moves

#
preds = [-1,0,1]
graph = {0: -1,
         1: 0,
         2: 1}

#put into graph structure

graph = {i:v for i,v in enumerate(preds)}
graph

dfs_paths_recursion(graph,0,2)

# keep a queue of nodes, pop off if we encounter them
count = 0

ALLPATH = []

# make a bunch of linked lists
# groups = length of longest linked list +1

all_nodes = []
for i,v in graph.items():
    nd = node(i,v)
    all_nodes.append(nd)


class node(object):

    def __init__(self,value, next):
        self.value = value
        self.next = next



def dfs_paths_recursion(graph,cur, goal, path=None):

    if path==None:
        path = []
        path.append(cur)

    for v in (graph[cur]-set(path)):
        if v==goal:
            ALLPATH.append(path + [v])
        else:
            dfs_paths_recursion(graph, v, goal, path+[v])



list_preds = [[-1,0,1],[1,-1,3,-1],[-1,-1,-1], [1,2,3,-1,-1]]

for p in list_preds:
    print(groups(p))


def groups(pred):
    l = [x for x in pred if x != -1]
    return max(len(set(l)),1)


# what if we put in dict ?

d = {}
for item in p:
    add_to_dict(d,item)

d

def add_to_dict(d, key):
    """
    Add sequence counts to dict 
    """
    if key in d.keys():
        d[key] += 1
    else:
        d[key] = 1
