# Given a list of non negative integers, arrange them such that they form the largest number.

from typing import List

testCase = [121,12]

class numKey(str):
    def __lt__(x, y):
        return x+y > y+x

class Solution:
    @staticmethod
    def largestNumber(nums: List[int]) -> str:
        result = ''.join(sorted(map(str, nums), key=numKey))
        return '0' if result[0] == '0' else result

print(Solution.largestNumber(testCase))