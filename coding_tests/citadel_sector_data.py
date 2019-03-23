# sector data
# argo Sat - 3-23-2019
#############################
import numpy as np


def fib(N):
    """
    calculate fibonacci number
    
    :type N: int
    :rtype: int
    """
    import numpy as np

    arr = np.zeros(N+1)
    if N <= 1:
        return N

    # initialize
    arr[0] = 0
    arr[1] = 1
    for i in np.arange(2, N+1):

        arr[i] = int(arr[i - 1]) + int(arr[i - 2])
        print('f({})={}'.format(i, arr[i]))

    return int(arr[N])


fib(4)

for i in np.arange(10):
    print('{} => fib = {}'.format(i, fib(i)))

np.zeros(1)


# how to do this in log(n) time ?
##################################
# https://www.geeksforgeeks.org/program-for-nth-fibonacci-number/

# submitted on leetcode
# https://leetcode.com/problems/fibonacci-number/
# Fn = {[(√5 + 1)/2] ^ n} / √5

3**2

def fib_close_form(n):
    return round(((np.sqrt(5) + 1)/2)**n/np.sqrt(5))


round(1.2)

[fib_close_form(x) for x in np.arange(10)]

fib_close_form(2)

######################

# check if string is palindrome
# submitted here
# https://leetcode.com/problems/valid-palindrome/

s = "A man, a plan, a canal: Panama"

def isPalindrome(s):
    """
    :type s: str
    :rtype: bool
    """

    clean_s = ''.join(ch for ch in s if ch.isalnum()).lower()


    p_start = 0
    p_end = len(clean_s)-1

    while(p_start < p_end):

        if clean_s[p_start] != clean_s[p_end]:
            return False

        p_start += 1
        p_end -= 1

    return True

isPalindrome("emme")












