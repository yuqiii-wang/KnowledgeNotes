#include "ThreadPool.hpp"
#include <iostream>

void fun1(int slp)
{
    std::cout << "Sleep for " << slp  << " secs" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(slp * 1000));
    std::cout << "Finished sleeping" << std::endl;
}

int main()
{
    try 
    {
        std::threadpool executor{ 50 };

        std::future<void> ff = executor.commit(fun1,5);
    }
    catch (std::exception& e) {
        std::cout << "some unhappy happened... " << std::this_thread::get_id() << e.what() << std::endl;
    }

    return 0;
}