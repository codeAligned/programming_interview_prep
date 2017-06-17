# Leet Code Mock Interviews
# 6-4-2017
# Alex Chao


# Two Sigma

# Power of 4

def power_of_4(num):

n = 8
power_of_2(32)

def power_of_2(n):
    return (n) & (n-1) == 0


def isPowerOfFour(num):
    """
    :type num: int
    :rtype: bool
    """
    if num == 0:
        return False
    else:
        while num > 1:
            if num % 4 != 0:
                return False
            num = num / 4
        return True



[isPowerOfFour(x) for x in range(17)]