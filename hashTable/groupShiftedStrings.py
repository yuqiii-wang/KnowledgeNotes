"""
Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:

"abc" -> "bcd" -> ... -> "xyz"
Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.

Example:

Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
Output: 
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]
"""

def groupStrings(str_list):
	dict_key = {}
	for word in str_list:
		temp_word = list(word)
		shf_num = ord(temp_word[0]) - ord('a')
		key_val = ''.join([(chr((ord(char)-shf_num+26)%26)) for char in word])
		if key_val in dict_key:
			dict_key[key_val].append(word)
		else:
			dict_key[key_val] = [word]

	arr_ret = []
	for each in dict_key:
		arr_ret.append(dict_key[each])

	return arr_ret


if __name__=="__main__":
	str_list = ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"]
	arr_ret = groupStrings(str_list)
	print(arr_ret)
