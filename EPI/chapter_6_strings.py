# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-15-2017
# Author: Alex H Chao
# ************************************************************************************

import numpy as np
from random import randint
from helper.helper import *


# ************************************************************************************
# 6.1 - Interconvert Strings and Integers
# ************************************************************************************

# ************************************************************************************
# 6.2 - Base Conversion - perform base conversion, input: string, int b1, int b2
# converts from base b1 to base b2
# ************************************************************************************

s = '265'
b1 = 10
b2 = 8



# ************************************************************************************
# 6.3 - Compute Spreadsheet column encoding - Input: "D", return 4, input "AA", return 27
# ************************************************************************************

# ************************************************************************************
# 6.4 - Replace each 'a' with 2 'd's, delete each etry containing a b
# ************************************************************************************

# ************************************************************************************
# 6.5 - Test Palindromicity - takes string, returns true is its a palindrome (SHOULD BE EASY)
# NOTE: you should ignore non characters
# "A man, a plan, a canal: Panama" is a palindrome.
# ************************************************************************************
# trick is to use 2 ptrs

#WORKS - 5-16-2017

def is_palindrome(s):
    start = 0
    end = len(s) - 1
    while start < end:
        while (not s[start].isalnum()) & (start < end):
            start += 1
        while (not s[end].isalnum()) & (start < end):
            end -= 1
        if s[start].lower() != s[end].lower():
            return False
        start += 1
        end -= 1
    return True

def is_char(c):
    return (122 >= ord(c) >= 48)

#is_palindrome('A man, a plan, a canal: Panama')
#is_palindrome('.,')


# ************************************************************************************
# 6.6 - Reverse all Words in a Sentence - reverse words in a string
# e.g. input: Alice Likes Bob, returns Bob Likes Alice (GS ASKED ME THIS)
# ************************************************************************************
# TO DO

# ************************************************************************************
# 6.7 - Compute All Mnemonics for a Phone Number
# Input: phone number as string, returns all possible character sequences for that number
# Hint: Use recursion
# ************************************************************************************


