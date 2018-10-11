"""
Given a sorted array, two integers k and x, find the k closest elements to x in the array. The result should also be sorted in ascending order. If there is a tie, the smaller elements are always preferred.

Example 1:

Input: [1,2,3,4,5], k=4, x=3
Output: [1,2,3,4]
Example 2:

Input: [1,2,3,4,5], k=4, x=-1
Output: [1,2,3,4]
Note:

The value k is positive and will always be smaller than the length of the sorted array.
Length of the given array is positive and will not exceed 10^4
Absolute value of elements in the array and x will not exceed 10^4
"""
def findKneibor(arr, k, x):
    left = 0
    right = len(arr) - 1
    while(left <= right):
        mid = (left + right) // 2
        if arr[mid] < x:
            left = mid + 1
        elif arr[mid] > x:
            right = mid - 1
        else:
            return mid
    # not found
    print(mid)
    return -1

if __name__=="__main__":
    num_list = [int(i) for i in range(10)]
    num_list.remove(6)
    k = 3
    x = 99
    result = findKneibor(num_list, k, x)
    print(num_list)
    print("k = " + str(k) + ';\t' + 'x = ' + str(x))
    print(result)