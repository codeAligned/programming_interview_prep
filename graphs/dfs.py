# Python - Graph Algos
# DFS
# http://stanford.edu/~kailaix/cgi-bin/b/index.php?controller=post&action=view&id_post=55
# =========================



def dfs(graph, start):
    visited = set()
    stack = [start]

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

def dfs_recursion(graph, node, visited = []):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs_recursion(graph,n, visited)
    return visited

ALLPATH = []

def dfs_paths_recursion(graph,cur, goal, path=None):

    if path==None:
        path = []
        path.append(cur)

    for v in (graph[cur]-set(path)):
        if v==goal:
            ALLPATH.append(path + [v])
        else:
            dfs_paths_recursion(graph, v, goal, path+[v])

if __name__ == '__main__':


    graph = {'A': set(['B', 'C']),
             'B': set(['A', 'D', 'E']),
             'C': set(['A', 'F']),
             'D': set(['B']),
             'E': set(['B', 'F']),
             'F': set(['C', 'E'])}

    dfs(graph,'B')