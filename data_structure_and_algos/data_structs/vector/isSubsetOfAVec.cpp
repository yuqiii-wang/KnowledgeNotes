// check if sorted vector `a` is a subset of sorted vector `b`
// e.g., [1,1,2,3] is a subset of [1,1,2,3,4]

#include <cstdlib>
#include <iostream>
#include <vector>
using namespace std;

bool isSubset(vector<int>& a, vector<int>& b)
{
    int countA = 1;
    int countB = 1;
    int idxB = 0;
    if (a.size() == 0) {
        return true;
    }

    for (int i = 1; i < a.size(); i++) {
        while (a[i-1] == a[i]) {
            countA++;
            i++;
        }

        while (a[i-1] == b[idxB++]) {
            if (idxB+1 == b.size()) {
                return false;
            }
            countB++;
            if (countA < countB) {
                continue;
            }
        }
        if (countB < countA) {
            return false;
        }
        if (a[i-1] < b[idxB] && a[i] > b[idxB]) {
            return false;
        }
        
        countA = 1;
        countB = 1;
    }
    return true;
}

int main() {
    string input("1,2,3;1,2,3,4");

    int sep = input.find(';');
    string aStr(input.substr(0,sep));
    string bStr(input.substr(sep+1));

    std::cout << aStr << std::endl;
    std::cout << bStr << std::endl;

    vector<int> a;
    a.reserve(10);
    vector<int> b;
    b.reserve(10);
    for (const auto elem : aStr) {
        if (elem != ',')
            a.push_back(atoi(&elem));
    }
    for (const auto elem : bStr) {
        if (elem != ',')
            b.push_back(atoi(&elem));
    }

    std::cout << isSubset(a, b) << std::endl;

    return 0;
}