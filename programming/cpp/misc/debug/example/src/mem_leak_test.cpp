#include <iostream>
#include <vector>
#include <thread>

std::vector<int*>* func() {
    std::vector<int*>* ptrVector = new std::vector<int*>(
        {new int[10]}
    );
    return ptrVector;
}

int main() {
    std::cout << "main" << std::endl;;
    std::vector<int*>* a = func();
    delete a;
    return 0;
}