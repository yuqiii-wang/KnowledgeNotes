"""
Given two strings s and t, determine if they are isomorphic.

Two strings are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.

Example 1:

Input: s = "egg", t = "add"
Output: true
Example 2:

Input: s = "foo", t = "bar"
Output: false
Example 3:

Input: s = "paper", t = "title"
Output: true
Note:
You may assume both s and t have the same length.
"""

def isomorphicStrings(s, t):
	if len(s) != len(t):
		return None

	letter_record = {}
	for i, letter in enumerate(s):
		letter_record[letter] = t[i]

	duplicate_t = ''
	for letter in s:
		duplicate_t += letter_record[letter]

	if duplicate_t == t:
		return True 
	else:
		return False

s = "paper"
t = "title"
print(isomorphicStrings(s, t))