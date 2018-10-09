"""
Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.
"""

def groupAnagrams(array):
	arr_key = {}
	for each_word in array:
		key_val = ''.join(sorted(list(each_word)))
		if key_val in arr_key:
			arr_key[key_val].append(each_word)
		else:
			arr_key[key_val] = [each_word]

	arr_ret = []
	for each in arr_key:
		arr_ret.append(arr_key[each])

	return arr_ret

if __name__ == '__main__':
	array = ["eat", "tea", "tan", "ate", "nat", "bat"]
	arr_ret = groupAnagrams(array)
	print(arr_ret)