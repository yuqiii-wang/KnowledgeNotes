"""
A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.
"""
def findPeakElem(num_list):
    left = 0
    num_len = len(num_list)
    right = num_len - 1
    if num_len < 3:
        if num_len == 1:
            return 0
        else:
            if num_list[0] > num_list[1]:
                return 0
            else:
                return 1

    while(left <= right):
        mid = (left + right) // 2
        if mid == num_len - 1:
            return mid
        elif num_list[mid] > num_list[mid+1] and num_list[mid-1] < num_list[mid]:
            return mid
        elif num_list[mid] > num_list[mid+1]:
            right = mid - 1
        elif num_list[mid] < num_list[mid+1]:
            left = mid + 1

if __name__=="__main__":
    input_list = [1,2,3,4,3]
    result = findPeakElem(input_list)
    print(result)