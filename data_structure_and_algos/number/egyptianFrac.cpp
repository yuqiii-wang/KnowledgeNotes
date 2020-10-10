// https://www.geeksforgeeks.org/greedy-algorithm-egyptian-fraction/

#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

class egyptionFrac{
private:
    void __findFrac(int n_r, int d_r);
public:
    vector<int> result;
    egyptionFrac(int n_r, int d_r); // constructor
    void printResults();
};

egyptionFrac::egyptionFrac(int n_r, int d_r){
    __findFrac(n_r, d_r);
}

void egyptionFrac::printResults(){
    vector<int>::iterator iter;
    for (iter = result.begin(); iter != result.end(); iter++)
        cout << "1/" << *iter << " + ";
    cout << "0/1" << endl;
}

void egyptionFrac::__findFrac(int n_r, int d_r){
    long double val = static_cast<long double>(n_r) / static_cast<long double>(d_r);
    long double i;
    while(val > 1e-4){
        for (i = 1; i < 1e+5; i++){
            if (1 / i < val){
                result.push_back(i);
                break;
            }
        }
        val = val - 1 / i;
    }
}

int main(){
    egyptionFrac* EF = new egyptionFrac(6, 14);
    EF->printResults();
    egyptionFrac* EF2 = new egyptionFrac(7, 19);
    EF2->printResults();
    egyptionFrac* EF3 = new egyptionFrac(2, 3);
    EF3->printResults();
    return 0;
}
