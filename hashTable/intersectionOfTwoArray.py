"""
Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

Note:
Each element in the result must be unique.
The result can be in any order.
"""
import numpy as np

def intersectionOfTwoArray(array1, array2):
	arraySet1 = set(array1)
	intersectionArray = []
	for i in array2:
		# this is doen in internal hash via method such as key % bucket to locate element,
		# hence regarding as O(1)
		if i in arraySet1:
			intersectionArray.append(i)

	return intersectionArray

if __name__ == '__main__':
	array1 = list(np.random.randint(10, size=5))
	array2 = list(np.random.randint(10, size=5))
	print(array1)
	print(array2)
	print(intersectionOfTwoArray(array1, array2))