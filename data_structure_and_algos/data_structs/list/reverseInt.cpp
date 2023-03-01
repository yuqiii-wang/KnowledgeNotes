#include <iostream>
#include <list>
#include <bits/stdc++.h>

/*
    Reverse an integer

    Example 1:
    Input: x = 123
    Output: 321

    Example 2:
    Input: x = -123
    Output: -321

    Example 3:
    Input: x = 120
    Output: 21
*/


class SolutionByMod {
public:

    // INT_MAX = 2147483647;
    // INT_MIN = -2147483648;

    static int reverse(int x) {
        int res = 0;
        while(x!=0) {
            // take the suffix
            int tmp = x%10;

            if (res>214748364 || (res==214748364 && tmp>7)) {
                return 0;
            }
            if (res<-214748364 || (res==-214748364 && tmp<-8)) {
                return 0;
            }

            res = res*10 + tmp;
            x /= 10;
        }
        return res;
    }
};

int main()
{
    std::cout << SolutionByMod::reverse(12345) << std::endl;
    std::cout << SolutionByMod::reverse(-2147483412) << std::endl;
    std::cout << SolutionByMod::reverse(-123) << std::endl;
    std::cout << SolutionByMod::reverse(1534236469) << std::endl;
    
    return 0;
}