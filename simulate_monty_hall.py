# simulate monty hall
from random import randint
import numpy as np


"""
- generate random number 0,1,2 (this will be the prize door)
- generate another random number (this is the door you choose)
- monty hall shows you one of the empty doors
- you then can choose to switch to the other one


"""
all_switch = 0.0
no_switch = 0.0
all_choices = set([0,1,2])
n = 10000

for i in range(n):
    prize = randint(0,2)
    you_pick = randint(0,2)

    my_list = [prize, you_pick]

    # pick random sample
    choices_for_monty = all_choices.difference(my_list)
    monty_pick = np.random.choice(list(choices_for_monty) )

    switch = all_choices.difference(set([you_pick, monty_pick]))

    if switch == set([prize]):
        all_switch += 1

    if you_pick == prize:
        no_switch += 1

print(all_switch/n, no_switch/n)




