_opening_map = { '(':')', '{':'}', '[':']' }
    _closing_map = { ')':'(', '}':'{', ']':'[' }

def braces(values):
    out = []
    for s in values:
        out.append(braces_helper(s))
    return out

def braces_helper(s):
    stack = []
    bracket_mapping = {"]":"[", "}":"{", ")":"("}
    for c in s:
        if c in bracket_mapping.values(): # its a bracket, add to our stack
            stack.append(c)
        elif c in bracket_mapping.keys(): # its an end bracket
            if stack == [] or bracket_mapping[c] != stack.pop():
                return 'NO'
        else:
            return 'NO'
    if stack == []:
        return 'YES'
    else:
        return 'NO'




values = ['[]','{]}']
braces(values)


# ==== merge strings
s1 = 'abc'
s2 = 'defg'


s3 = zip(s1,s2)
''.join(s3)

flatten = lambda l: [item for sublist in l for item in sublist]

_s = [item for sublist in s3 for item in sublist]
''.join(_s)

import itertools
''.join([item for sublist in itertools.izip_longest(s1,s2) for item in sublist])




# stupid way
a = 'abc'
b = 'defg'
combined = []
i=0
while (i < len(a)) & (i < len(b)):
    combined.append(a[i])
    combined.append(b[i])
    i += 1

while i < len(a):
    combined.append(a[i])
    i +=1

while i < len(b):
    combined.append(b[i])
    i +=1

''.join(combined)

def foo(n):
    for x in range(n):
        yield x**3

for x in foo(5):
    print(x)

# === lets try again

def balanced(s):
    dic = { '{':'}',
            '[':']',
            '(':')'
            }




