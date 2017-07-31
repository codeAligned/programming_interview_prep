# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 7-4-2017 Happy 4th of July!
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np


# ************************************************************************************
# 18.1 - Search a Maze
# good online resources
# http://bryukh.com/labyrinth-algorithms/
# http://www.geeksforgeeks.org/shortest-path-in-a-binary-maze/
# ************************************************************************************

mat  = [          [1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
                  [1, 0, 1, 0, 1, 1, 1, 0, 1, 1 ],
                  [1, 1, 1, 0, 1, 1, 0, 1, 0, 1 ],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 1 ],
                  [1, 1, 1, 0, 1, 1, 1, 0, 1, 0 ],
                  [1, 0, 1, 1, 1, 1, 0, 1, 0, 0 ],
                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1 ],
                  [1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ],
                  [1, 1, 0, 0, 0, 0, 1, 0, 0, 1 ]]


maze  = [         [0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 1, 0, 1, 0],
                  [0, 0, 1, 1, 0]]


Source = [0, 0]
Destination = [3, 4]


# use BFS or DFS,
# BFS has advanatage that it is the shortest path

# found online
"""
def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    # lets add all the open spaces (ie if maze[i][j] == 0)
    graph = {(i, j): [] for j in range(width) for i in range(height)
             if not maze[i][j]}
    for row, col in graph.keys():
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(("S", (row + 1, col)))
            graph[(row + 1, col)].append(("N", (row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(("E", (row, col + 1)))
            graph[(row, col + 1)].append(("W", (row, col)))
    return graph
"""

# my own version - removed N,S,E,W
def maze2graph(maze):
    height = len(maze)
    width = len(maze[0]) if height else 0
    # lets add all the open spaces (ie if maze[i][j] == 0)
    graph = {(i, j): [] for j in range(width) for i in range(height)
             if not maze[i][j]}
    for row, col in graph.keys(): # for each open spot
        if row < height - 1 and not maze[row + 1][col]:
            graph[(row, col)].append(((row + 1, col)))
            graph[(row + 1, col)].append(((row, col)))
        if col < width - 1 and not maze[row][col + 1]:
            graph[(row, col)].append(((row, col + 1)))
            graph[(row, col + 1)].append(((row, col)))
    return graph

graph = maze2graph(mat)

# now that we have a graph, apply BFS or DFS

def dfs_paths_recursion(graph,cur, goal, path=None):

    if path==None:
        path = []
        path.append(cur)

    for v in (set(graph[cur])-set(path)):
        print('v = {}, goal = {}, path = {}'.format(v,goal, path))
        if v==goal:
            out_path = list()
            out_path.append(v)
            print('path found! path is {}'.format(out_path))
            return out_path
        else:
            dfs_paths_recursion(graph, v, goal, path+[v])

# not sure why it doesnt return the path
p = dfs_paths_recursion(graph, cur=(0,0), goal=(2,0),path=None)
print(p)



# BFS
# ===

# found online
# https://pythoninwonderland.wordpress.com/2017/03/18/how-to-implement-breadth-first-search-in-python/
# finds shortest path between 2 nodes of a graph using BFS
def bfs_shortest_path(graph, start, goal):
    # keep track of explored nodes
    visited = []
    # keep track of all the paths to be checked
    queue = [[start]]

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while queue:
        # pop the first path from the queue
        path = queue.pop(0)
        # get the last node from the path
        node = path[-1]
        if node not in visited:
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # mark node as explored
            visited.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("

bfs_shortest_path(graph, start=(0,0),goal=(3,1))



