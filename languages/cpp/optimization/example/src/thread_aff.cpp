#include <thread>
#include <mutex>
#include <iostream>
#include <vector>

void testNativeHandle()
{
    std::mutex iomutex;
    std::thread t = std::thread([&iomutex] {
        {
        std::lock_guard<std::mutex> iolock(iomutex);
        std::cout << "Thread: my id = " << std::this_thread::get_id() << "\n"
                    << "        my pthread id = " << pthread_self() << "\n";
        }
    });

    {
        std::lock_guard<std::mutex> iolock(iomutex);
        std::cout << "Launched t: id = " << t.get_id() << "\n"
                << "            native_handle = " << t.native_handle() << "\n";
    }

    t.join();
}

void testCpuAffinity()
{
    constexpr unsigned num_threads = 4;
  // A mutex ensures orderly access to std::cout from multiple threads.
  std::mutex iomutex;
  std::vector<std::thread> threads(num_threads);
  for (unsigned i = 0; i < num_threads; ++i) {
    threads[i] = std::thread([&iomutex, i] {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      int cnt = 0;
      while (cnt++ < 3) { // run for 3 secs
        {
          // Use a lexical scope and lock_guard to safely lock the mutex only
          // for the duration of std::cout usage.
          std::lock_guard<std::mutex> iolock(iomutex);
          std::cout << "Thread #" << i << ": on CPU " << sched_getcpu() << "\n";
        }

        // Simulate important work done by the tread by sleeping for a bit...
        std::this_thread::sleep_for(std::chrono::milliseconds(900));
      }
    });

    // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
    // only CPU i as set.
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(i, &cpuset);
    int rc = pthread_setaffinity_np(threads[i].native_handle(),
                                    sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
      std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n";
    }
  }

  for (auto& t : threads) {
    t.join();
  }
}

int main()
{
    testNativeHandle();

    testCpuAffinity();

    return 0;
}