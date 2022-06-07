"""
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example:

Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version.
"""
def isBadVersion(input_version, whichBadVersion=0):
    if input_version >= whichBadVersion:
        return True
    else:
        return False

def findBadVersion(totalVersionNum):
    left = 0
    right = totalVersionNum
    
    while(left <= right):
        mid = (left + right) // 2
        if mid == 0 and isBadVersion(mid):
            return mid
        elif isBadVersion(mid) and not isBadVersion(mid-1):
            return mid
        elif isBadVersion(mid):
            right = mid - 1
        else:
            left = mid + 1
    return -1

if __name__=="__main__":
    totalVersionNum = 10
    result = findBadVersion(totalVersionNum)
    print(result)