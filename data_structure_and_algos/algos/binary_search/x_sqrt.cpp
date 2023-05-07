// find the square root of a number x
// x's square root solution should exist between 0 and x itself.
// binary search updates upper bound `ub` and lower bound `lb` until `ub` and `lb` meet each other

#include <iostream>

class Solution {
public:
    static int mySqrt(int x) {

        if ( x == 1) {
            return 1;
        }

        float ub = x;
        float lb = 0;
        float mid = 0;
        while (ub - lb > 0.01) {
            mid = (ub - lb) / 2.0 + lb;
            if ( (double) mid * (double)mid > x) {
                if ( (long)mid * (long)mid == x ) {
                    return (int)mid;
                }
                ub = mid;
            } else if ( (double) mid * (double) mid < x) {
                if ( (long)mid * (long)mid == x ) {
                    return (int)mid;
                }
                lb = mid;
            }
            else {
                return (int)mid;
            }
        }
        
        return (int)mid;
    }

    static int mySqrtPureInt(int x) {
        if ( x == 1) {
            return 1;
        }

        float ub = x;
        float lb = 0;
        float ans = 0;
        while (lb <= ub) {
            int mid = (ub - lb) / 2 + lb;
            if ( (long)mid * mid <= x ) {
                ans = mid;
                lb = mid + 1;
            }
            else {
                ub = mid - 1;
            }
        }
        return ans;
    }

    static int mySqrtByAssembly(int x){
        double s = x;
        __asm__ (
                "movq %1, %%xmm0;"
                "sqrtsd %%xmm0, %%xmm1;"
                "movq %%xmm1, %0"
                :"+r"(s)
                :"r"(s)
                );
        return s;
    }
};

int main()
{
    int result = Solution::mySqrt(9);
    int result2 = Solution::mySqrtPureInt(9);
    int result3 = Solution::mySqrtByAssembly(9);
    std::cout << result << std::endl;
    std::cout << result2 << std::endl;
    std::cout << result3 << std::endl;

    return 0;
}