// fibonacci with traditional recursion and dynamic programming


#include <iostream>
#include <unordered_map>

using namespace std;

int fib_recursion(int idx){
    if (idx == 1 || idx == 2)
        return 1;
    return fib_recursion(idx-1) + fib_recursion(idx-2);
}

int fib_dynamicProg(int idx, unordered_map<int, int> mem){
    if (mem.find(idx) != mem.end())
        return mem[idx];
    
    if (idx == 1 || idx == 2)
        return 1;
    
    int result = fib_dynamicProg(idx-1, mem) + fib_dynamicProg(idx-2, mem);
    mem[idx] = result;
    
    return result;
}

int main(){
    unordered_map<int, int> mem;
    cout << fib_dynamicProg(7, mem) << endl;
    cout << fib_recursion(7) << endl;
    return 0;
}
