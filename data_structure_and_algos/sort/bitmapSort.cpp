// bitmap sort is a memory consuming sort algorithm but time complexity is O(n)
// bascially, it just records the indices of elements in a bit array so that when
// iterating the bit array, elements come in a sorted order.
// One important constratint of bitmap sort is high cardinality that there should be
// no duplicate elements


#include <bitset>
#include <vector>
#include <iostream>

#define MAX_VAL 100 // estimate the max val in a array that should not exceed MAX_VAL as defiend here

using namespace std;

vector<int> bitmapSort(vector<int> arr){
    
    vector<int> sortedArr;
    bitset<MAX_VAL + 1> bitArr;
    
    for (int i = 0; i < arr.size(); i++)
        bitArr.set(arr[i]);
    
    for (int i = 0; i < MAX_VAL+1; i++)
        if (bitArr.test(i))
            sortedArr.push_back(i);
    
    return sortedArr;
}

int main (){
    vector<int> arr = {5,1,2,13,7,10,0,20,16,9};
    
    vector<int> sortrArr = bitmapSort(arr);
    
    for (int i = 0; i < sortrArr.size(); i++)
        cout << sortrArr[i] << " ";
    cout << endl;
    
    return 0;
 }
     
