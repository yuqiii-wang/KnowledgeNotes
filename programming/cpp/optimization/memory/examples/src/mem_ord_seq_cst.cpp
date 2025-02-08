#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
 

// Both defined in thread 1 and thread 2
std::atomic<int> x (0);
std::atomic<int> y (0);
int r1, r2, r3;

void th1() {
    x.store(1, std::memory_order_seq_cst); // A
    y.store(1, std::memory_order_release); // B
}

void th2() {
    r1 = y.fetch_add(1, std::memory_order_seq_cst); // C
    r2 = y.load(std::memory_order_relaxed); // D
}

void th3() {
    y.store(3, std::memory_order_seq_cst); // E
    r3 = x.load(std::memory_order_seq_cst); // F
}

int main() {

    for (int i = 0; i < 10; ++i)
    {
        // run synchronously
        std::thread a(th1);
        std::thread b(th2);
        std::thread c(th3);

        a.join();
        b.join();
        c.join();

        std::cout << "r1: " << r1 
            << ", r2: " << r2
            << ", r3: " << r3
            << std::endl;
    }

    return 0;
}
