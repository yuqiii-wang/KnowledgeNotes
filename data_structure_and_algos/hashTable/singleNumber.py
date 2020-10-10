"""
Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
"""
import numpy as np
from functools import reduce

def singleNumber(array):
	result = reduce(lambda a, b: a ^ b, array)
	if result is 0:
		return None
	return result

if __name__ == '__main__':
	array = [3,3,1,2,1,5,2,6,6] # every element appears twice except for one.
	print(str(array))
	print(singleNumber(array))