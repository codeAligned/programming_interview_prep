# found online
# https://github.com/vj-ug/Python-Graphs
# http://stanford.edu/~kailaix/cgi-bin/b/index.php?controller=post&action=view&id_post=55


def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def shortest_path(graph, start, goal):
    try:
        return next(bfs_paths(graph, start, goal))
    except StopIteration:
        return None






def bfs_paths_iteration(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)

        for next in graph[vertex] - set(path):
            if next == goal:
                # print(path+[next])
                yield path + [next]
            else:
                queue.append((next, path + [next]))

ALLPATH = []
def bfs_paths_recursion(graph, cur, goal, path=None):
    if path==None:
        path = []
        path.append(cur)

    for v in (graph[cur]-set(path)):
        if v==goal:
            ALLPATH.append(path + [v])
            #return path + [v]
        else:
            bfs_paths_recursion(graph, v, goal, path+[v])



if __name__ == '__main__':


    graph = {'A': set(['B', 'C']),
             'B': set(['A', 'D', 'E']),
             'C': set(['A', 'F']),
             'D': set(['B']),
             'E': set(['B', 'F']),
             'F': set(['C', 'E'])}

    bfs_paths_recursion(graph, 'F','D')
