#include <vector>
#include <iostream>
#include <thread>
#include <atomic>
 
std::atomic<int> cnt = {0};
int cnt_noAtomic = {0};
 
void f()
{
    for (int n = 0; n < 1000; ++n) {
        cnt.fetch_add(1, std::memory_order_relaxed);
    }
}

void f_noAtomic()
{
    for (int n = 0; n < 1000; ++n) {
        cnt_noAtomic++;
    }
}
 
int main()
{
    std::vector<std::thread> v;
    std::vector<std::thread> v_noAtomic;
    for (int n = 0; n < 10; ++n) {
        v.emplace_back(f);
        v_noAtomic.emplace_back(f_noAtomic);
    }

    for (auto& t : v) {
        t.join();
    }
    for (auto& t : v_noAtomic) {
        t.join();
    }
    
    std::cout << "Final counter value is " << cnt << '\n';
    std::cout << "Final counter value (noAtomic) is " << cnt_noAtomic << '\n';
}