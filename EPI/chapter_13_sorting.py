# ************************************************************************************
# INTERVIEW PREP - Programming Interview Questions from
# Elements of Programming Interviews
# Created 5-29-2017 - Mon
# Author: Alex H Chao
# CH 13 - Sorting
# ************************************************************************************
import math as m
import numpy as np
from helper.helper import *
from random import randint
# ************************************************************************************
# 13 - Given list of events, return the maximum number of overlapping events
# ************************************************************************************
# Methodology:
#  - sort all event endpoints and tag start or end to each endpoint
#  - have a counter -> increment counter for every start, decrement for every end
#  - e.g. 1 - start, 2 - start, 4 - start, 5 - end, 5 - end, 6 = start, 7 - end
#  - counter = [1,2,3,2,1,2,1]
#  - sorting takes nlogn and iterating takes another n