#include <iostream>

int main() {
    std::cout << "main\n";
    int * a = new int[10]; // mem leak for no delete a
    return 0;
}