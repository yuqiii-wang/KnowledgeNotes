
// atomic::load/store example
#include <iostream>       // std::cout
#include <atomic>         // std::atomic, std::memory_order_relaxed
#include <vector>
#include <thread>         // std::thread

// `__builtin_expect` for branch prediction to speed up if condition execution
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)

const int numThreads = 1000;
const int newFoo = 99;

// later compare foo vs fooNoAtomic
std::atomic<int> foo (0);
int fooNoAtomic = 0;

std::atomic<int> fooNoAtomicDiffCount (0);
std::atomic<int> fooDiffCount (0);

void set_foo(int x) {
  foo.store(x,std::memory_order_relaxed);     // set value atomically
}

void print_foo() {
  int x;
  do {
    x = foo.load(std::memory_order_relaxed);  // get value atomically
  } while (x==0);

  if (unlikely(x != newFoo))
    fooDiffCount++;
}

void set_foo_noAtomic(int x) {
  fooNoAtomic = x;
}

void print_foo_noAtomic() {
  if (unlikely(fooNoAtomic != newFoo))
    fooNoAtomicDiffCount++;
}

int main ()
{  
  std::vector<std::thread> print_foo_thread_vec;
  std::vector<std::thread> set_foo_thread_vec;
  print_foo_thread_vec.reserve(numThreads);
  set_foo_thread_vec.reserve(numThreads);

  for (int i = 0; i < numThreads; ++i) {
    print_foo_thread_vec.push_back(std::thread(print_foo)); 
    set_foo_thread_vec.push_back(std::thread(set_foo, newFoo)); 
  }

  for (auto& thread : print_foo_thread_vec) {
    thread.join();
  }
  for (auto& thread : set_foo_thread_vec) {
    thread.join();
  }

  std::vector<std::thread> print_fooNoAtomic_thread_vec;
  std::vector<std::thread> set_fooNoAtomic_thread_vec;
  print_fooNoAtomic_thread_vec.reserve(numThreads);
  set_fooNoAtomic_thread_vec.reserve(numThreads);
  for (int i = 0; i < numThreads; ++i) {
    print_fooNoAtomic_thread_vec.push_back(std::thread(print_foo_noAtomic)); 
    set_fooNoAtomic_thread_vec.push_back(std::thread(set_foo_noAtomic, newFoo)); 
  }

  for (auto& thread : print_fooNoAtomic_thread_vec) {
    thread.join();
  }
  for (auto& thread : set_fooNoAtomic_thread_vec) {
    thread.join();
  }

  std::cout << "fooNoAtomicDiffCount: " << fooNoAtomicDiffCount << std::endl;
  std::cout << "fooDiffCount: " << fooDiffCount << std::endl;

  return 0;
}