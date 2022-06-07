"""
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
"""

def arrSrch(num_list, tgt):
    if not num_list: # if empty return -1
        return -1

    num_len = len(num_list)
    if num_len == 1:
        if num_list[0] == tgt:
            return 0
        else:
            return -1

    num_rotate = num_list.index(min(num_list))
    new_list = num_list[num_rotate:]
    new_list.extend(num_list[:num_rotate])

    left = 0
    right = num_len
    while(left <= right):
        mid = (left + right) // 2
        try:
            if new_list[mid] == tgt:
                return (mid + num_rotate) % num_len
            elif new_list[mid] > tgt:
                right = mid - 1
            else:
                left = mid + 1
        except:
            return -1

    return -1

if __name__=="__main__":
    nums = [3, 1, 2]
    tgt = 5
    result = arrSrch(nums, tgt)
    print(result)