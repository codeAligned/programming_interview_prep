# Brain Bench test
# 6-26-2017


def countdown_from_n(n):
    while n >= 0:
        yield n
        n -= 1


c = countdown_from_n(10)


l = []

import datetime
dt = datetime.datetime.now()

import datetime
datetime.now()


tup = ('a','b','c','d','e')
tup.delete('c')
tup[-4]
a = ['1','2','3']
[a1,a2,a3] = a


def id(it):
    i = range(len(it))
    for i,v in zip(i,it):
        yield i,v

l = [x*2 for x in range(10)]
x = id(l)
x.next()

for i,v in iter()

a,b = 23, 17
quot = a//b
remain = a - (int(quot) *b)
quotrem = (quot, remain)

quotrem
divmod(23,17)

(div(23,17), mod(23,17))

l = ['a','b','c','d','e']
l[2:-1]

str = "iterator"
it = iter(str)
it.
print(it.next())
print(it.next())
print(it.next())
print(list(it))

a = int(1)
b = float(2.0)
c = complex(3)

x = a + b + c
type(x)


k = [1,2,3]
v = ['a','b','c']

d = dict(k,v)

d = dict()
for x in range(len(k)):
    d[k[x]] = v[x]

x = 1,2,3
a,b,c = 1,2,3
d,e = 1,2,3


print("the %d dogs", (1+2))
str = 'the {0} dogs'
print(str.format(1+2))


a = [
    [1,2,3],
    [4,5,6],
    [7,8,9]
]


a


b = [[row[i] for row in a] for i in range(len(a))]
b


def g(): g.s = 'something'
print(g.__dict__)
print(g.s)
g()
print(g.s)
print(g.__dict__)


import re
def f1(data):
    p = re.compile('(?P<dept>[A-Z]{2,3]) (?P<num>[0-9]{3})')
    return p.search(data)


obj = f1('CS 101')
d, n = obj['dept'], obj['num']
obj.group('dept')


a = 4
b= 7

x = lambda: a if 1 else b
lambda x: 'big' if x > 100 else 'small'
print(x())

import string

pat = 'ab'
s = 'abb'


s.find(s[s.find(pat)+1:],pat)
string.index(s[string.index(s,pat):],pat)

0 <= s.find(pat) < s.rfind(pat)

class Person(object):
    def __init__(self, name):
        print(name)


class Bob(Person):
    def __init__(self, name):
        print('bob')

class Sue(Person):
    def __init__(self, name):
        print('sue')

class Child(Bob,Sue):
    def __init__(self, name):
        print('sue')

class Dog(object):
    def speak(self):
        print('woof')

class chi(object):
    def speak(self):
        print('yip')
        super(self).speak()

gunther = chi()
gunther.speak()


func(1,2,3)
func()

def func(*param, **kw):
    print(param,kw)




s1 = {1,2,3,5,8}
s2 = {2,3,5,7,11}

print(sorted(s1.union(s2)))
print(sorted(s1.intersection(s2)))
print(sorted(s1.difference(s2)))
print(sorted(s1.symmetric_difference(s2)))



def x():
    print("X")
    return True

def y():
    print("Y")
    return False

def z():
    if x() or y(): print("X or Y")
    if x() and y(): print("X and Y")
    else: print("Z")

z()

from functools import reduce
x = lambda x: reduce(lambda y,z: y*z, range(1,x+1))
print(x(5))



from multiprocessing import Process, Pipe

def f(conn):
    conn.send('This is sent thru a pip!')
    conn.close()

def g(conn):
    conn.send('This is sent thru a pipe too!')
    conn.close()


parent_conn, child1_conn = Pipe()
parent_conn, child2_conn = Pipe()
p = Process(target=f, args={child1_conn,})

q = Process(target=g, args={child2_conn,})
q.start()
p.start()
print(parent_conn.recv())
p.join()
q.join()






