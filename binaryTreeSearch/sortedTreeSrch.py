"""
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
"""
def search(nums, tgt):
    if len(nums) == 1:
        return 0 if nums[0] == tgt else -1
    split_half = len(nums) // 2
    if nums[:split_half][-1] > tgt:
        return search(nums[:split_half], tgt)
    elif nums[:split_half][-1] < tgt:
        ret_idx = search(nums[split_half:], tgt)
        return ret_idx if ret_idx == -1 else ret_idx + split_half
    else:
        return split_half - 1

if __name__=="__main__":
    nums = [-1,0,3,5,9,12]
    tgt = 9
    ret_idx = search(nums, tgt)
    print(ret_idx)
