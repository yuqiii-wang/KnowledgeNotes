// permutation, used -std=c++11

#include <iostream>
#include <vector>

using namespace std;

class permutation{
    vector<char> &arr;
    vector<char>* resultPermutation;
    long int arrLen;
    long int total_result;
    long int count = 0;

    private:
        void do_permutation(int left, int right);
        long int do_calculateTotalResults(long int arrLen);
    
    public:
        permutation(vector<char> &arr); // constructor
        void printResults(); // show results
        ~permutation(); // destructor
};

permutation::permutation(vector<char> &arr):arr(arr){
    arrLen = arr.size();
    long int total_result = do_calculateTotalResults(arrLen);
    this->total_result = total_result;
    resultPermutation = new vector<char>[total_result];
    do_permutation(0, arrLen - 1);
}

long int permutation::do_calculateTotalResults(long int arrLen){
    if (arrLen > 1)
        return do_calculateTotalResults(arrLen - 1) * arrLen;
    else
        return 1;
}

void permutation::do_permutation(int left, int right){
    if (left == right){
        resultPermutation[count] = arr;
        count++;
    } else {
        for (int i = left; i <= right; i++){
            swap(arr[left], arr[i]);
            do_permutation(left+1, right);
            swap(arr[left], arr[i]);
        }
    }
}

void permutation::printResults(){
    vector<char>::iterator iter;
    for (int idx = 0; idx < this->total_result; idx++){
        cout << '[';
        for(iter = resultPermutation[idx].begin(); iter != resultPermutation[idx].end(); iter++)
            cout << *iter << ' ';
        cout << "] " << idx << endl;
    }
}

permutation::~permutation(){
    delete [] resultPermutation;
}

int main(){
    vector<char> arr = {'1', '2', '3', '4', '5'};

    permutation* P = new permutation(arr);
    P->printResults();

    return 0;
}