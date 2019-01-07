#include <iostream>

using namespace std;

int main(){
    void *a;
    a = new int(1);
    cout << a << endl;
    cout << *(int *)a << endl;
}