"""
Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the result is returned.
"""
def sqrtBinTree(num):
    # to speed up computation by starting at half of num
    left = 0
    right = num

    while(left <= right):
        mid = (left + right) // 2
        if mid ** 2 == num: 
            return mid
        elif mid ** 2 > num:
            right = mid - 1
        else:
            left = mid + 1

    return left - 1

if __name__=="__main__":
    num = 82
    ret = sqrtBinTree(num)
    print(ret)