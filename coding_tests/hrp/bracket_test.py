# harried allied coding test
# 8/6 sunday

# function to test if brackets are balanced


def balanced(s):
    dic = {'{': '}',
           '[': ']',
           '(': ')'
           }
    stack = []
    for char in s:

        if char in dic.keys():  # we have a left parenthesis
            stack.append(dic[char])  # add to stack the right side

        elif char in dic.values():  # we have a right parenth
            if (len(stack) == 0) or (stack.pop() != char):
                return False
    return len(stack) == 0

balanced('{[}]')

def test():
    if balanced('{[}]') == False:
        print('test1 passed')
    if balanced('{[]}') == True:
        print('test1 passed')

test()


# =======
# merge 2 strings so that they are in alternating order

import itertools
from itertools import izip_longest

a = 'aaaaaaa'
b = 'bbbbb'

def merge_strings(a,b):
    dif = len(a) - len(b)
    c = zip(a,b)
    out = list(itertools.chain(*c)) + list(b[dif:]) if dif < 0 else list(itertools.chain(*c)) + list(a[-dif:])
    return ''.join(out)


def merge_strings_1_liner(a,b):
    return ''.join([y for y in itertools.chain(*[
    x for x in izip_longest(a,b)]) if y is not None])

merge_strings('aaaa','bbb')
merge_strings_1_liner('aa','bbb')