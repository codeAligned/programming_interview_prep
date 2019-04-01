# https://www.hackerrank.com/interview/interview-preparation-kit
# 3-29-2019
####################

####################
# Dynamic Programming
####################

# Minimum Swaps 2
# https://www.hackerrank.com/challenges/minimum-swaps-2/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
#You are given an unordered array consisting of consecutive integers  [1, 2, 3, ..., n]
# without any duplicates. You are allowed to swap any two elements. You need to find the
# minimum number of swaps required to sort the array in ascending order.
#################

# - if int is in correct place, move on
# - else, swap it to the "correct" place

def minimumSwaps(arr):
    swap, i = 0,0

    while i < len(arr):
        if arr[i] == i+1 :# its already in the correct spot
            i += 1
            continue
        else: # else arr[i] is in the wrong place, where does it belong? arr[i]+1
            correct_index = arr[i]-1

            arr[correct_index ], arr[i] = arr[i], arr[correct_index ]
            swap += 1
        #print(arr)
    return swap

arr = [4,3,1,2]
minimumSwaps(arr)


# Array Manipulation
# https://www.hackerrank.com/challenges/crush/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=arrays
# similar to "Render a Calendar" in EPI 13.5 p 190

#################

# brute force solution
def arrayManipulation(n, queries):
    print(n, queries)
    arr = n * [0]

    for query in queries:
        start, end, amt = query[0]-1, query[1], query[2]
        for i in range(start,end):
            arr[i] += amt
        print(arr)

    return max(arr)

# best one found online
def arrayManipulation(n, queries):
    N, M = n, len(queries)
    arr = [0 for i in range(N + 2)]
    for query in queries:
        a, b, k = query[0], query[1], query[2]
        arr[a] += k
        arr[b + 1] -= k

    for i in range(1, len(arr)):
        arr[i] += arr[i - 1]
    return max(arr)


n = 5
queries =  [[1, 2, 100],
            [2, 5, 100]]

arrayManipulation(n, queries)


N, M = n, 3
arr = [0 for i in range(N+2)]
for query in queries:
    a,b,k = query[0], query[1], query[2]
    arr[a] += k
    arr[b + 1] -= k
    print(arr)

for i in range(1, len(arr)):
    arr[i] += arr[i-1]
print(max(arr))



# New Year Chaos
# https://www.hackerrank.com/challenges/new-year-chaos/problem
# u have a queue or people, each person can bribe person in front max of 2 times
# count total number of bribes (or "Too chaotic" if someone bribed >2 times
##############################################


def minimumBribes(Q):
    #
    # initialize the number of moves
    moves = 0
    #
    # decrease Q by 1 to make index-matching more intuitive
    # so that our values go from 0 to N-1, just like our
    # indices.  (Not necessary but makes it easier to
    # understand.)
    Q = [P-1 for P in Q]
    #
    # Loop through each person (P) in the queue (Q)
    for i,P in enumerate(Q):
        # i is the current position of P, while P is the
        # original position of P.
        #
        # First check if any P is more than two ahead of
        # its original position
        if P - i > 2:
            print("Too chaotic")
            return
        #
        # From here on out, we don't care if P has moved
        # forwards, it is better to count how many times
        # P has RECEIVED a bribe, by looking at who is
        # ahead of P.  P's original position is the value
        # of P.
        # Anyone who bribed P cannot get to higher than
        # one position in front if P's original position,
        # so we need to look from one position in front
        # of P's original position to one in front of P's
        # current position, and see how many of those
        # positions in Q contain a number large than P.
        # In other words we will look from P-1 to i-1,
        # which in Python is range(P-1,i-1+1), or simply
        # range(P-1,i).  To make sure we don't try an
        # index less than zero, replace P-1 with
        # max(P-1,0)
        for j in range(max(P-1,0),i):
            if Q[j] > P:
                moves += 1
    print(moves)

# New Year Chaos
# https://www.hackerrank.com/challenges/new-year-chaos/problem
# u have a queue or people, each person can bribe person in front max of 2 times
# count total number of bribes (or "Too chaotic" if someone bribed >2 times
##############################################
#Its not exactly a fast solution but I used this same idea. Here is my python code
# for it. First I generate a dictionary with a k,v pair which is (length of substring,
# substrings of the same length). Then I generate all the possible substrings and for each
# one I sort them and them find out if the sorted substring is already in the array of sorted
# substrings of the same length. If so we've found another anagram so we increment the
# anagram counter. After generating all the substrings we will also have the number of anagrams.


def sherlockAndAnagrams(s):
    anagrams = 0
    substrings = {i: [] for i in range(1, len(s) + 1)}
    for i in range(0, len(s)):
        for j in range(i+1, len(s) + 1):
            substr = " ".join(sorted(s[i:j]))
            for k in range(0, len(substrings[j-i])):
                if substrings[j-i][k] == substr:
                    anagrams += 1
            substrings[j-i].append(substr)
    return anagrams



####################
# Greedy Algos
####################

arr = [10,102,20,30,100,101]

maxMin(k=3,arr=arr)

def maxMin(k, arr):
    arr.sort()
    total_min = max(arr)

    i = 0
    while i <len(arr)-k:
        subarr = arr[i:(i+k)]
        min_max = max(subarr)-min(subarr)
        total_min = min(total_min, min_max)
        i +=1

    return total_min

####################
# Search
####################


# Ice Cream Parlor

# leverage the 2sum function
def test_three_sum():
    a = [11,2,5,7,3]
    a.sort()
    print(two_sum(a,22))
    print(three_sum(a, 22))

# assume sorted
def two_sum(a, num):
    # 2 ptrs
    start = 0
    end = len(a)-1

    while start < end: # tricky, make sure its < not <=
        this_sum = a[start] + a[end]
        if this_sum == num:
            #we found it
            return True
        elif this_sum < num:
            start += 1
        else:
            end -= 1
    return False


def three_sum(a, target):
    # check if two_sum = target - current num
    a.sort()
    for x in a:
        found = two_sum(a, target - x)
        print(x, target-x)
        if found:
            return True
    return False

def whatFlavors(a, num):
    a.sort()

    # 2 ptrs
    start = 0
    end = len(a)-1

    while start < end: # tricky, make sure its < not <=
        this_sum = a[start] + a[end]
        if this_sum == num:
            #we found it
            return("{} {}".format(start+1, end+1))
        elif this_sum < num:
            start += 1
        else:
            end -= 1
    return False


whatFlavors(a = [1, 4, 5, 3, 2], num=5)

################
# Pairs (like 2sum but 2diff)

arr = [1, 5, 3, 4, 2]
ans = 3 #4-2, 5-3, 3-1

# PASSED ALL TESTS!
def pairs(k, arr):
    d = {}
    for i,x in enumerate(arr):
        if x not in d:
            d[x] = 1
    count = 0
    for i,x in enumerate(arr):
        diff = k+x
        try:
            if d[diff] == 1:
                count += 1
        except:
            continue
    return count




####################
# Dynamic Programming
####################

####################
# Max Array sum
# https://www.hackerrank.com/challenges/max-array-sum/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
# Given an array of integers, find the subset of non-adjacent elements with the maximum sum.
# Calculate the sum of that subset.
# For example, given an array [-2,1,3,-4,5] -> 8
####################
arr = [-2,1,3,-4,5]

max_array_sum_non_adj(arr)

def max_array_sum_non_adj(arr):
    max_so_far = [0]* len(arr)
    max_so_far[0] = arr[0]
    max_so_far[1] = max(arr[0], arr[1])

    for i in range(2,len(arr)):
        #print(i,a)
        # max of (a[i], max_so_far[i-1], max_so_far[i-2]+a[i])
        max_so_far[i] = max([arr[i],
                 max_so_far[i-1],
                 max_so_far[i-2] + arr[i]])
        print("i={}, arr[i]={}, max_so_far={}".format(i, a, max_so_far))
    return max_so_far[-1]


# Abbreviation
# string matching but capitalization / removal of letters allowed
s1 = 'AaFeb'
s2 = 'AFE'
abbreviation(s1,s2)

abbreviation(s1,s2, verbose = True)

# still not working

def abbreviation(s1,s2, verbose=True):
    """
    Given 2 strings, compute the edit distance of the 2 strings
    Parameters
    ----------
    s1
    s2

    Returns
    -------

    """
    s1 = ' ' + s1 # add space to begining
    s2 = ' ' + s2
    A = [[0 for col in range(len(s1))] for row in range(len(s2))]

    rows = len(s2)
    cols = len(s1)

    A[0][0] = 0
    #fill out first row and col 1st
    #for i in range(0,rows):
    #    A[i][0] = 1 # copy sum down

    for j in range(0,cols):
        A[0][j] = 1

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            #if UPPER case match or if S1
            #print("row={}, col={}".format(row,col))
            if ((s2[row].upper() == s1[col].upper()) and (A[row-1][col]==1)):
                A[row][col] = 1
            elif s1[col].isupper():
                A[row][col] = 0# s1[col] is upper case then set to false
                #print(A)
            else: # copy over the last row
                A[row][col] = A[row][col-1]
            #print(A)

    if verbose:
        return A
    else:
        return 'YES' if A[rows-1][cols-1]==1 else 'NO'

######################
# found online, but not sure if it works

def abbreviation(a,b):
    canDo = True
    b = list(b)
    for j in a:
        if j.isupper():
            if j in b:
                b.remove(j)
            else:
                canDo = False

    if len(b) == 0 and canDo:
        return("YES")
    else:
        return("NO")

#####################
# Candies
# https://www.hackerrank.com/challenges/candies/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming

######################

#

arr = [4, 6, 4, 5, 6, 2]

arr= [9, 2, 3, 6, 5, 4, 3, 2, 2, 2]

if len(arr) == 1:
    return [1]

candies = [1] * len(arr)

i=1
while i < len(arr):
    if arr[i] > arr[i-1]:
        candies[i] = candies[i-1] + 1
    i += 1

candies

candies(n=0, arr=arr)

def candies(n, arr):
    if len(arr) == 1:
        return [1]

    candies_1 = [1] * len(arr)
    candies_2 = [1] * len(arr)

    i = 1
    while i < len(arr):
        if arr[i] > arr[i - 1]:
            candies_1[i] = candies_1[i - 1] + 1
        i += 1

    i=1
    while i > 0:
        if (arr[i] > arr[i + 1]) and (candies[i] <= candies[i+1]):
            candies_1[i] = candies_1[i + 1] + 1
        i -= 1

    return candies_1


# solution found
def candies(n, arr):
    count = [1]
    for i,x in enumerate(arr[1:],1):
        if x <= arr[i-1]:
            count.append(1)
        else:
            count.append(count[i-1]+1)
    for i,x in enumerate(arr[::-1],2):
        if x <= arr[n-i+1]:
            count[n-i] = max(count[n-i], 1)
        else:
            count[n-i] = max(count[n-i], count[n-i+1]+1)
    return sum(count)




####################
# Graphs
####################



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

