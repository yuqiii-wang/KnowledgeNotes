"""
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
"""

def twoSum(array, target):
	# for hash map __find_a_key() is considered O(1)
	array = set(array)
	for elem in array:
		if target - elem in array:
			reslt_elem_1 = elem
			reslt_elem_2 = target - elem
			return reslt_elem_1, reslt_elem_2

	return None

array = [2, 7, 11, 15]
target = 9
print(twoSum(array, target))