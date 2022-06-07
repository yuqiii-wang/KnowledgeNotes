"""
We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number is higher or lower.

-1 : My number is lower
 1 : My number is higher
 0 : Congrats! You got it!
"""
def guess(input_guess, pick=1):
    if input_guess < pick:
        return 1
    elif input_guess > pick:
        return -1
    else:
        return 0

def guessNum(num):
    left = 0
    right = num
    while(left <= right):
        mid = (left + right) // 2
        result = guess(mid)
        if result == 0:
            return mid
        elif result == -1:
            right = mid - 1
        else:
            left = mid + 1

if __name__=="__main__":
    num = 1239923456789
    result = guessNum(num)
    print(result)