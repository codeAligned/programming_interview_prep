# equilibrium index of an array
import numpy as np

  A[1] =  3
  A[2] = -4
  A[3] =  5
  A[4] =  1
  A[5] = -6
  A[6] =  2
  A[7] =  1

A = [-1,3,-4,5,1,-6,2,1]

for i in range(1, len(A)-1):
    print sum(A[:i])
    print sum(A[i+1:])
    if sum(A[:i]) == sum(A[i+1:]):
        print i

#base case
if len(A) <= 1:
    return -1
# more efficient solution
total_sum = np.sum(A)
left_sum = 0# initialize
for i in range(0,len(A)):
    total_sum = total_sum - A[i]
    print 'i = ', i, ': left_sum = ', left_sum, '; total_sum = ', total_sum
    left_sum = left_sum + A[i]
return -1

def solution(A):
    """
    A = array
    """
    # write your code in Python 2.7
    for i in range(1,len(A)):
        if sum(A[:i]) == sum(A[i + 1:]):
            return i


# much better solution
def solution(A):
    if len(A) <= 0:
        return -1
    # more efficient solution
    total_sum = np.sum(A)
    left_sum = 0# initialize
    for i in range(0,len(A)):
        total_sum = total_sum - A[i]
        #print 'i = ', i, ': left_sum = ', left_sum, '; total_sum = ', total_sum
        if total_sum == left_sum:
            return i
        left_sum = left_sum + A[i]
    return -1



#1) fibonacci

def fib(n):
    saved_fib = [0, 1]
    for i in range(2, n + 1):
        saved_fib.append(saved_fib[i - 1] + saved_fib[i - 2])
    return saved_fib[n]


for i in range(15):
    print fib(i)


#- print fibonacci numbers fib(n) until its > x
# compare fib(n) vs x and fib(n-1) vs x
# return the minimum

# modified fib to return both fib(n) and fib(n-1)
def fib_mod(n):
    saved_fib = [0, 1]
    for i in range(2, n + 1):
        saved_fib.append(saved_fib[i - 1] + saved_fib[i - 2])
    return saved_fib[n], saved_fib[n-1]

x = 15
n=1
last_num,fib_n_minus_1 = 0,0
list_fibs = []

while last_num < x:
    last_num = fib(n)
    list_fibs.append(last_num)
    n += 1

print n, list_fibs
list_fibs[-1]
list_fibs[-2]

def fib(n):
    saved_fib = [0, 1]
    for i in range(2, n + 1):
        saved_fib.append(saved_fib[i - 1] + saved_fib[i - 2])
    return saved_fib[n]

def func(x):
    n = 1
    last_num = 0
    list_fibs = []

    while last_num < x:
        last_num = fib(n)
        list_fibs.append(last_num)
        n += 1

    if len(list_fibs) < 2:
        return 0
    return min(abs(list_fibs[-1]-x), abs(list_fibs[-2]-x))


func(25)


#5) how many substrings of all the same character?


S = 'zzzyz'

my_dict = {}
for i in S:
    my_dict[i] = 1


for i, letter in enumerate(S):
    if my_dict.has_key(letter):
        my_dict[letter] += 1
    else:
        my_dict[letter] = 1

total = 0
for k,v in my_dict.items():
    total += 2 * v-1
total


#4) student scores

A = [1,6,3,4,3,5]

a,b,c = 0,1,2

# 3 ptrs
if len(A) <3:
    return A


#new array
while A != B:

    B = A[:]
    for a in range(len(A)-2):
        b=a+1
        c=a+2
        if (A[a] > A[b]) & (A[c] > A[b]): # increment
            B[b] += 1
        elif (A[a] < A[b]) & (A[c] < A[b]):
            B[b] -= 1

if A == B:
    return B
else:
    func(B)


def func(A):
    # 3 ptrs
    if len(A) < 3:
        return A

    B = A[:]

    while B_first != B:
        B_first = B[:]
        for a in range(len(A) - 2):
            b = a + 1
            c = a + 2
            if (A[a] > A[b]) & (A[c] > A[b]):  # increment
                B[b] += 1
            elif (A[a] < A[b]) & (A[c] < A[b]):
                B[b] -= 1


B = A[:]
B_first = []
while B_first != B:
    B_first = B[:]
    for a in range(len(A) - 2):
        b = a + 1
        c = a + 2
        if (A[a] > A[b]) & (A[c] > A[b]):  # increment
            B[b] += 1
        elif (A[a] < A[b]) & (A[c] < A[b]):
            B[b] -= 1



def take_test(A):
    B = A[:]
    for a in range(len(A) - 2):
        b = a + 1
        c = a + 2
        if (A[a] > A[b]) & (A[c] > A[b]):  # increment
            B[b] += 1
        elif (A[a] < A[b]) & (A[c] < A[b]):
            B[b] -= 1
    return B


x = 0
while x == 0:
    B = take_test(A)
    B2 = take_test(B)
    A = B[:]
    print B, B2
    if B == B2:
        break

B = A[:]
while A != B:
    A = B[:]
    B = take_test(A)
B


# 3---- hexspeak

S = '257'
hex_num = hex(int(S)).split('x')[1]
h_copy = hex_num[:]
for c in 'ABCDEF10':
    hex_num.replace(c,'', inplace=True)


hex_num = hex(int(S)).split('x')[1]
for c,d in zip(['0','1'],['O','l']):
    hex_num.replace(c,d)
