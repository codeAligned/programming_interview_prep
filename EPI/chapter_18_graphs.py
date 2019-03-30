# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 7-4-2017 Happy 4th of July!
# Author: Alex H Chao
# ************************************************************************************
import math as m
import numpy as np
from collections import deque

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

maze  = [         [0, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]]

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

graph = maze2graph(maze)

# now that we have a graph, apply BFS or DFS

def dfs_paths_recursion(graph,cur, goal, path=None, verbose = False):

    if path==None:
        path = []
        path.append(cur) # keep track of ur path

    for v in (set(graph[cur])-set(path)): # only consider new paths
        if verbose:
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


###############
# Revisit 3-30-2019
# https://backtobackswe.com/the-fundamentals
###############
# BFS and DFS are the same except DFS uses STACK, BFS uses QUEUE!
# really? lets test it out


def dfs_shortest_path_iterative(graph, start, goal):
    """
    

    Parameters
    ----------
    graph
    start
    goal

    Returns
    -------

    """
    # keep track of explored nodes
    visited = []
    # keep track of all the paths to be checked
    stack = [[start]]

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while stack:
        # pop the first path from the queue
        path = stack.pop()
        # get the last node from the path
        node = path[-1]
        if node not in visited:
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                stack.append(new_path)
                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # mark node as explored
            visited.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("


#######
shortest_path(graph, start=(0,0), goal=(0,2),method='DFS')
shortest_path(graph, start=(0,0), goal=(0,2),method='BFS')

######

def shortest_path(graph,
                  start,
                  goal,
                  method = 'DFS',
                  verbose = True):
    """
    
    Parameters
    ----------
    graph
    start
    goal
    method - BFS or DFS

    Returns
    -------

    """
    # keep track of explored nodes
    visited = []
    # keep track of all the paths to be checked
    stack = [[start]]

    # return path if start is goal
    if start == goal:
        return "That was easy! Start = goal"

    # keeps looping until all possible paths have been checked
    while stack:
        # pop the first path from the queue
        if method == 'DFS':
            path = stack.pop() # stack
        else:
            path = stack.pop(0) # queue
        # get the last node from the path
        node = path[-1]
        if node not in visited:
            neighbours = graph[node]
            # go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                stack.append(new_path)

                if verbose:
                    print('path={}, visited={}, node = {}, neighbours = {}, next_path = {}'.format(
                            path,  visited, node,          neighbours, new_path))

                # return path if neighbour is goal
                if neighbour == goal:
                    return new_path

            # mark node as explored
            visited.append(node)

    # in case there's no path between the 2 nodes
    return "So sorry, but a connecting path doesn't exist :("


##############
# 18.2 Paint a boolean matrix (minesweeper) or under solns is "matrix_connected_regions.py"
##############

import collections

def flip_color(x, y, image):

    color = image[x][y]
    q = collections.deque([(x, y)])
    image[x][y] = 1 - image[x][y]  # Flips.
    while q:
        x, y = q.popleft()
        for next_x, next_y in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            if (0 <= next_x < len(image) and 0 <= next_y < len(image[next_x])
                    and image[next_x][next_y] == color):
                # Flips the color.
                image[next_x][next_y] = 1 - image[next_x][next_y]
                q.append((next_x, next_y))


def flip_color_wrapper(x, y, image):
    flip_color(x, y, image)
    return image


image = [[1, 1, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 1, 0],
        [ 1, 0, 0, 0]]

flip_color_wrapper(0,0,image)


# hacker rank
###############
# get biggest region
# DFS: Connected Cell in a Grid
# https://www.hackerrank.com/challenges/ctci-connected-cell-in-a-grid/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=graphs

#https://www.hackerrank.com/challenges/ctci-connected-cell-in-a-grid/forum?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=graphs

# - for row in rows, col in columns
# - if value =1
# - getregionsize(matrix, row, col)
# -  this does a DFS

def getBiggestRegion(grid):
    maxRegion = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            maxRegion = max(maxRegion, countCells(grid, row, col))
    return maxRegion


def countCells(grid, row, col):
    if (not (row in range(len(grid)) and col in range(len(grid[0])))):
        return 0 # if out of range..
    if (grid[row][col] == 0):
        return 0
    count = 1
    grid[row][col] = 0
    count += countCells(grid, row + 1, col)
    count += countCells(grid, row - 1, col)
    count += countCells(grid, row, col + 1)
    count += countCells(grid, row, col - 1)
    count += countCells(grid, row + 1, col + 1)
    count += countCells(grid, row - 1, col - 1)
    count += countCells(grid, row - 1, col + 1)
    count += countCells(grid, row + 1, col - 1)
    return count

########################

image = [[0, 0, 1 ,1],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [1, 1, 0, 0]]

getBiggestRegion(grid=image)
















