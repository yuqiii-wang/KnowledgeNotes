// https://www.geeksforgeeks.org/minimum-sum-absolute-difference-pairs-two-arrays/

// notes: pair means two elems from diff arrays are paired; elems cannot repeatedly selected
// the min sum should be sum of elem gaps of two sorted arrays'. Only when sorted the gap between each elem is minimal.

#include <iostream>
#include <vector>

using namespace std;

int minDiff(vector<int> a, vector<int> b){
    sort(a.begin(), a.end());
    sort(b.begin(), b.end());
    
    int result = 0;
    for (int i = 0; i < a.size(); i++)
        result += abs(a[i] - b[i]);
    
    return result;
}

int main(){
    cout << minDiff({2,3,4}, {1,4,5}) << endl;
    return 0;
}
