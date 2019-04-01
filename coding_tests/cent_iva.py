
# Sunday 3-31-2019
##################
# 4:19 -> 7:19

# q 1
##################

def countHoles(num):
    # Write your code here
    def count_holes_per_digit(d):
        if d == 8:
            return 2
        elif d in [0,4,6,9]:
            return 1
        else:
            return 0

    for i in num:

num = 1288

for i,x in enumerate(str(num)):
    print(i,x)

##################
# q 2 distinct pairs


1288

arr = [1,2,3,6,7,8,9,1]


###


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

############
# nice!

def numberOfPairs(a, k): #k = target, a is array
    # Write your code here
    d = {}
    a.sort()

    start = 0
    end = len(a)-1
    count = 0

    while start < end:
        this_sum = a[start] + a[end]
        if this_sum == k:
            # add pair to our dict
            this_set = str(set([a[start],a[end]]))
            if this_set not in d.keys():
                count += 1
                #found new unique pair
                d[this_set] = 1
            start +-1
            end -=1
        elif this_sum < k:
            start += 1
        else:
            end -= 1
    return count


a = [1,2,3,6,7,8,9,1]
k = 10
numberOfPairs(a,k)

start = 0
end=3

set([a[start],a[end]])

##


##### buy / sell stock for max profit
def maxProfit(self, prices):
    profit = 0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            profit += diff
    return profit


price = [3,4,5,3,5,2]

def buy_and_sell_stock_k_times(prices, k):

    if not k:
        return 0.0
    elif 2 * k >= len(prices):
        return sum(max(0, b - a) for a, b in zip(prices[:-1], prices[1:]))
    min_prices, max_profits = [float('inf')] * k, [0] * k
    for price in prices:
        for i in reversed(list(range(k))):
            max_profits[i] = max(max_profits[i], price - min_prices[i])
            min_prices[i] = min(min_prices[i],
                                price - (0 if i == 0 else max_profits[i - 1]))
    return max_profits[-1]


def buy_and_sell_stock_k_times_ac(prices, k):
    min_prices, max_profits = [float('inf')] * k, [0] * k
    for price in prices:
        for i in reversed(list(range(k))):
            max_profits[i] = max(max_profits[i], price - min_prices[i])
            min_prices[i] = min(min_prices[i],
                                price - (0 if i == 0 else max_profits[i - 1]))
    return max_profits[-1], max_profits

buy_and_sell_stock_k_times_ac(prices = price, k=5)

max_so_far = 0
running_max = [0]*len(price)

price_reversed = price[::-1]

for i,p in enumerate(price_reversed):
    running_max[i] = max(max_so_far, p)
    max_so_far = max(max_so_far, p)

sum([max(x-y,0) for x,y in zip(running_max[::-1],price)])

price[::-1]

### working!!!!
def maximumProfit(price):
    # Write your code here
    max_so_far = 0
    running_max = [0]*len(price)

    price_reversed = price[::-1]

    for i,p in enumerate(price_reversed):
        running_max[i] = max(max_so_far, p)
        max_so_far = max(max_so_far, p)

    return sum([max(x-y,0) for x,y in zip(running_max[::-1],price)])



## q6 - username disparity
#submitted

def usernameDisparity(inputs):
    def longest_prefix(a, b):
        count = 0
        min_len = min(len(a), len(b))

        for i in range(min_len):
            if a[i] == b[i]:
                count += 1
            else:
                break
        return count

    # inputs = str(inputs)
    # inputs = "".join(e for e in inputs if e.isalpha())
    inputs = inputs[0]
    # print("inputs={}".format(inputs))
    counts = len(inputs)

    if counts == 0:
        return [0]

    for i in range(1, len(inputs)):
        substr = inputs[i:]
        counts += longest_prefix(inputs, substr)
        # print('counts={}, substr={}'.format(counts,substr))
    return [counts]



###
def longest_common_subsequence(a,b):
    """
    Returns the longest common subsequence
    Parameters
    ----------
    a = string
    b = string

    Returns
    -------
    number (or Matrix)
    """
    a, b = list(a), list(b)
    n, m = len(a), len(b)
    M = initialize_matrix_zeros(n+1,m+1)

    # remember to -1 to indexing of a,b

    # start at row, col = 1
    lcs = []
    for row in range(1,n+1):
        for col in range(1,m+1): # m+1 since we added a "null" col
            if a[row-1] == b[col-1]: # subtract one since we add null col
                #print(a[row-1],b[col-1])
                lcs.append(a[row-1])
                M[row][col] = 1 + max( M[row-1][col], M[row][col-1] )
                # letters equal, increment LCS
            else:
                M[row][col] = max(M[row - 1][col], M[row][col - 1])
                # else, carry over

    return M, lcs


longest_common_subsequence(a='ababa', b = 'ababa')

######


a = 'ababa'


count = len(a)

for i in range(1,len(a)):
    substr = a[i:]
    count += longest_prefix(a, substr)



def longest_prefix(a,b,):
    count = 0
    min_len = min(len(a), len(b))

    for i in range(min_len):
        if a[i] == b[i]:
            count+=1
        else:
            break
    return count


longest_prefix(a,a[4:])

# longest prefix common to both strings


def usernameDisparity(inputs):
    def longest_prefix(a, b):
        count = 0
        min_len = min(len(a), len(b))

        for i in range(min_len):
            if a[i] == b[i]:
                count += 1
            else:
                break
        return count

    inputs = str(inputs)
    count = len(inputs)

    for i in range(1, len(inputs)):
        substr = inputs[i:]
        count += longest_prefix(inputs, substr)
    return count

usernameDisparity(inputs = 'ababa')

inputs = ['aa']

inputs[0]

### q7 -> anagram diff
#### didnt get all cases to work
### submitted

def how_many_chars_in_common(s, t):
    s_d = {}
    for letter in list(set(s)):
        s_d[letter] = s.count(letter)

    count = 0
    for letter in list(set(t)):
        try:
            count += s_d[letter]
        except:
            continue
    return count


def min_chars_to_change_to_make_anagrams(a, b):
    count = how_many_chars_in_common(a, b)
    return max(len(a), len(b)) - count


def getMinimumDifference(a, b):
    # print("a={}".format(a))
    # print("b={}".format(b))

    out = []
    for i in range(len(a)):
        _a = a[i]
        _b = b[i]
        if len(_a) != len(_b):
            out.append(-1)
        else:
            out.append(min_chars_to_change_to_make_anagrams(_a, _b))
            # print(out)
    return out


###3
def edit_distance(s1,s2, verbose=True):
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
    A = initialize_matrix_zeros(len(s1), len(s2))

    rows = len(s1)
    cols = len(s2)

    A[0][0] = 0
    #fill out first row and col 1st
    for i in range(1,rows):
        A[i][0] = A[i-1][0] + 1 # copy sum down

    for j in range(1,cols):
        A[0][j] = A[0][j-1] + 1

    #now fill in the rest of the matrix
    for row in range(1,rows):
        for col in range(1,cols):
            #TRICK: if no edit is needed, just copy it over
            if s1[row] == s2[col]:
                A[row][col] = A[row-1][col-1]
            else: # we need to either replace, add, or remove
                A[row][col] = 1 + min(A[row-1][col],
                                      A[row][col-1],
                                      A[row-1][col-1])

    if verbose:
        return A
    else:
        return A[rows-1][cols-1]


edit_distance(s1 = 'aa',s2 = 'bb', verbose=True)

####3
isAnagram(s= 'abcd', t='cdaba')


def isAnagram(s, t):
    if len(s) != len(t):
        return False
    for letter in list(set(s)):
        if s.count(letter) != t.count(letter):
            return False
    return True

s = 'abbd'
t = 'abb'

s_d = {}
for letter in list(set(s)):
    s_d[letter] = s.count(letter)

count = 0
for letter in list(set(t)):
    try:
        count += s_d[letter]
    except:
        continue
count

s_d

min_chars_to_change_to_make_anagrams(s,t)

def min_chars_to_change_to_make_anagrams(a,b):
    count = how_many_chars_in_common(a,b)
    return max(len(a), len(b))-count

def how_many_chars_in_common(a,b):
    s_d = {}
    for letter in list(set(s)):
        s_d[letter] = s.count(letter)

    count = 0
    for letter in list(set(t)):
        try:
            count += s_d[letter]
        except:
            continue
    return count
#########

def how_many_chars_in_common(s,t):
    s_d = {}
    for letter in list(set(s)):
        s_d[letter] = s.count(letter)

    count = 0
    for letter in list(set(t)):
        try:
            count += s_d[letter]
        except:
            continue
    return count

def min_chars_to_change_to_make_anagrams(a,b):
    count = how_many_chars_in_common(a,b)
    return max(len(a), len(b))-count


def getMinimumDifference(a, b):
    # Write your code here
    str_a = a[0]
    str_b = b[0]
    #print(str_a,str_b)
    if len(str_a) != len(str_b):
        return -1
    else:
        return min_chars_to_change_to_make_anagrams(str_a, str_b)


getMinimumDifference(a=['aa'], b = ['bb'])


# s8 - palindromatic substrings
###############################
# works!
def countPalindromes(s):
    # Write your code here
    n = len(s)
    if n == 0:
        return 0
    arr = [1]*n
    for i in range(n):
        for j in range(i+1,n):
            if (s[j] == s[i]) and (s[i:j+1] == s[i:j+1][::-1]):
                arr[i] += 1
    return sum(arr)


def countSubstrings(s):
    """
    :type s: str
    :rtype: int
    """
    n = len(s)
    if n == 0:
        return 0
    arr = [1]*n
    for i in range(n):
        for j in range(i+1, n):
            if (s[j] == s[i]) and (s[i:j+1] == s[i:j+1][::-1]):
                arr[i] += 1
    return sum(arr)

countSubstrings('aaa')

n=3
arr = [1 for i in range(n)]


###########    q 4 - zombie clusters
import collections

def flip_color(x, y, image):

    color = image[x][y]
    q = collections.deque([(x, y)])
    #image[x][y] = 1 - image[x][y]  # Flips.
    while q:
        x, y = q.popleft()
        for next_x, next_y in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)):
            if (0 <= next_x < len(image) and 0 <= next_y < len(image[next_x])
                    and image[next_x][next_y] == color):
                # Flips the color.
                #image[next_x][next_y] = 1 - image[next_x][next_y]
                q.append((next_x, next_y))
            print(q)



flip_color(0,0,image)




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

def countRegion(grid, row, col):
    if (not (row in range(len(grid)) and col in range(len(grid[0])))):
        return 0 # if out of range..
    if (grid[row][col] == 0):
        return 0
    #count = 1
    #grid[row][col] = 0
    countCells(grid, row + 1, col)
    countCells(grid, row - 1, col)
    countCells(grid, row, col + 1)
    countCells(grid, row, col - 1)
    countCells(grid, row + 1, col + 1)
    countCells(grid, row - 1, col - 1)
    countCells(grid, row - 1, col + 1)
    countCells(grid, row + 1, col - 1)
    return 1


def countRegion(grid, row, col):
    if (not (row in range(len(grid)) and col in range(len(grid[0])))):
        return 0 # if out of range..
    if (grid[row][col] == 0):
        return 0
    #count = 1
    grid[row][col] = 0
    countRegion(grid, row + 1, col)
    countRegion(grid, row - 1, col)
    countRegion(grid, row, col + 1)
    countRegion(grid, row, col - 1)
    #countRegion(grid, row + 1, col + 1)
    #countRegion(grid, row - 1, col - 1)
    #countRegion(grid, row - 1, col + 1)
    #countRegion(grid, row + 1, col - 1)
    return 1


########################
input = ['1100', '1110', '0110', '0001']

input_mat = [list(x) for x in input]

map(int,list('1100'))

[[int(x) for x in list(y)] for y in input]

map(int, input_mat)

input_mat

image = [[0, 0, 1 ,1],
         [0, 0, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 1, 0, 1]]

image = [[1, 1, 0 ,0],
         [1, 1, 1, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 1]]

getBiggestRegion(grid=image)

countRegions(grid=[[0]])

image =[0]

if len(image)==0:
    return 0

def countRegions(grid):
    count = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            count += countRegion(grid, row, col)
    return count

# best email addr regex so far

#  '[a-z][^0-9]{1,6}_{0,1}[0-9]{0,4}(@hackerrank.com)'
'[a-z]{1,6}_{0,1}[0-9]{0,4}(@hackerrank.com)'





