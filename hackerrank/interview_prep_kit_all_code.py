# https://www.hackerrank.com/interview/interview-preparation-kit
# 3-29-2019
####################


####################
# Dynamic Programming
####################

####################
# Max Array sum
# https://www.hackerrank.com/challenges/max-array-sum/problem?h_l=interview&playlist_slugs%5B%5D=interview-preparation-kit&playlist_slugs%5B%5D=dynamic-programming
# Given an array of integers, find the subset of non-adjacent elements with the maximum sum.
# Calculate the sum of that subset.
# For example, given an array [-2,1,3,-4,5] -> 8
####################
arr = [-2,1,3,-4,5]

max_array_sum_non_adj(arr)

def max_array_sum_non_adj(arr):
    max_so_far = [0]* len(arr)
    max_so_far[0] = arr[0]
    max_so_far[1] = max(arr[0], arr[1])

    for i in range(2,len(arr)):
        #print(i,a)
        # max of (a[i], max_so_far[i-1], max_so_far[i-2]+a[i])
        max_so_far[i] = max([arr[i],
                 max_so_far[i-1],
                 max_so_far[i-2] + arr[i]])
        print("i={}, arr[i]={}, max_so_far={}".format(i, a, max_so_far))
    return max_so_far[-1]




####################


