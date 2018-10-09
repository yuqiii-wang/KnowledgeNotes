"""
Given an array of integers, find if the array contains any duplicates.

Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
"""
import numpy as np

class containDuplicate:

	def __init__(self, array):
		self.bool_ret = self.containDuplicate(array)

	def containDuplicate(self, array):
		return len(set(array)) != len(array)

	def __str__(self):
		return str(self.bool_ret)

if __name__ == '__main__':
	array = list(np.random.randint(1000, size=10))
	print(containDuplicate(array))
	array = list(np.random.randint(1, size=10))
	print(containDuplicate(array))