# Diamond Mine
# Start at (0,0), go to (n-1, n-1), collecting diamonds on the way
# After reaching (n-1,n-1), find a path back to (0,0)
# Return the maximum number of diamonds collected

M = [[1,3,5,8],[4,2,1,7],[4,3,2,3]]
len(M[0])
len(M)
calc_min_cost_path(M)

def calc_min_cost_path(M):
    """
    input an m by n matrix M of costs,
    return the min cost from (0,0) to (m,n)
    """
    A = initialize_matrix_zeros(3, 4)

    rows = len(M)
    cols = len(M[0])

    A[0][0] = M[0][0]
    #fill out first row and col 1st
    for i in range(1,rows):
        A[i][0] = A[i-1][0] + M[i][0] # copy sum down

    for j in range(1,cols):
        A[0][j] = M[0][j] + A[0][j-1]

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            A[row][col] = M[row][col] + min(A[row-1][col], A[row][col-1])
    return A[rows-1][cols-1]


def initialize_matrix_zeros(rows,cols):
    return [[0 for x in range(cols)] for y in range(rows)]



# Same but
# - with obstacles (-1)
# - diamonds (we want to maximize money)
# - when we reach bottom right, we can go back to 0,0

def calc_max_profit(M):
    """
    input an m by n matrix M of costs,
    return the min cost from (0,0) to (m,n)
    """
    rows = len(M)
    cols = len(M[0])

    A = [[0 for x in range(cols)] for y in range(rows)]
    #M_NEW = A.copy()

    # fill out first row and col 1st
    for i in range(1, rows):
        if (M[i][0] == -1) | (A[i - 1][0] == -1):
            A[i][0] = -1 # copy barries
        else:
            A[i][0] = A[i - 1][0] + M[i][0]

    for j in range(1, cols):
        if (M[0][j] == -1) | (A[0][j - 1] == -1):
            A[0][j] = -1 # copy barries
        else:
            A[0][j] = M[0][j] + A[0][j - 1]

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            if (M[row-1][col] == -1) & (M[row][col-1] == -1): # both left and top unreachable
                A[row][col] = -1
            elif M[row-1][col] == -1: # top is unreachable
                A[row][col] = M[row][col] + A[row][col-1]
            elif M[row][col-1] == -1:  # left is unreachable
                A[row][col] = M[row][col] + A[row-1][col]
            else: # current coin value + max of upper or left
                A[row][col] = M[row][col] + max(A[row - 1][col], A[row][col - 1])

    return A


M = [[0,1,-1],[1,0,-1],[1,1,1]]
M = [[0,1,1],[1,0,1],[1,1,1]]

calc_max_profit(M)

