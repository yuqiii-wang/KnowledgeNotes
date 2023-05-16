// find the max number below an input number `n`, the max number's digits are chosen from a set of one-digit numbers
// n = 24384, set = {2,3,9}, the max number is 23999.

#include <iostream>
#include <vector>
#include <set>

using namespace std;

int findMaxNumBelowInputNumFromSet(int num, set<int>& numSet) {
    int maxFinalNum = 0;
    int remain = num % 10;
    num = num / 10;

    vector<int> remainVec;
    remainVec.reserve(10);
    remainVec.push_back(remain);

    while ( num != 0 ) { 
        remain = num % 10;
        num = num / 10;
        remainVec.push_back(remain);
    }

    int maxDigit = 0;
    for (auto& elem : numSet) {
        maxDigit = max(elem, maxDigit);
    }
    
    bool shouldFillWithMax = false;
    for (int j = remainVec.size()-1; j >= 0; j--) {
        if (shouldFillWithMax) {
            remainVec[j] = maxDigit;
            continue;
        }
        int maxElem = 0;
        auto elem = numSet.find(remainVec[j]);
        if ( elem != numSet.end() ) {
            remainVec[j] = *elem;
        }
        else {
            for (auto& elem : numSet) {
                if (elem < remainVec[j]) {
                    maxElem = max(elem, maxElem);
                }
            }
            shouldFillWithMax = true;
            remainVec[j] = maxElem;
        }
    }

    int carryDigit = 1;
    for (int i = 0; i < remainVec.size(); i++) {
        maxFinalNum += carryDigit * remainVec[i];
        carryDigit *= 10;
    }

    return maxFinalNum;
}

int main() {
    string inputStr{"24384;2,3,9"};

    int numIdx = inputStr.find_first_of(';');
    string numStr = inputStr.substr(0,numIdx);
    int num = atoi(numStr.c_str());
    std::cout << num << std::endl;

    string vecStr = inputStr.substr(numIdx);
    set<int> setDigits;
    for (int i = 1; i < vecStr.size(); i++) { // should start from the next char
        if (vecStr[i] != ',') {
            setDigits.insert(atoi(&vecStr[i]));
        }
    }
    for (auto& elem : setDigits)
    std::cout << elem << "," << std::endl;

    std::cout << findMaxNumBelowInputNumFromSet(num, setDigits) << std::endl;

    return 0;
}