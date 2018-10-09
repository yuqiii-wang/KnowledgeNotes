"""
Write an algorithm to determine if a number is "happy".

A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.

Example: 

Input: 19
Output: true
Explanation: 
1^2 + 9^2 = 82
8^2 + 2^2 = 68
6^2 + 8^2 = 100
1^2 + 0^2 + 0^2 = 1
"""

import numpy as np 

# Python includes a data type for sets. A set is an unordered collection with no duplicate elements. 

def happyNumber(num):
	# the loop should run forever until it meets same calculated results (indicates that it is False)
	if num is 0:
		return False
	if num is 1:
		return True 

	result_set = set()

	while(1):
		result = 0
		while(num is not 0):
			result += (num%10)**2
			num = int(num/10)

		if result not in result_set:
			result_set.add(result)
		else:
			return False

		if result is 1:
			return True

		num = result

num = 19
print(num)
print(happyNumber(num))