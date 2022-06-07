"""
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
"""
def twoSum(nums, target):
    if not nums:
        return 

    count = 0
    while(nums):
        num1 = target - nums[0]
        nums.pop(0)
        if num1 in nums:
            return [count, nums.index(num1) + count + 1]
        else:
            count += 1

if __name__=="__main__":
    num_list = [int(i) for i in range(10)]
    num_list.remove(6)
    tgt = 3
    result = twoSum(num_list, tgt)
    print(num_list)
    print("tgt = " + str(tgt))
    print(result)