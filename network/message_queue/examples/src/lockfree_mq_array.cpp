#include<vector>
#include<atomic>
#include<thread>
#include<assert.h>

#include<unordered_map>
#include<mutex>
#include<condition_variable>
#include<thread>
#include<iostream>
#include <ctime>

using namespace std;


template<typename T>
class LockFreeArrayQueue {
public:
	enum class Strategy {
		ABANDON, 
		FORCE,  
		YIELD,  // std::this_thread::yield
	};
public:
	// construction should finish in one thread only, since it should be static shared by different threads
	LockFreeArrayQueue(int capacity):_data(capacity), _capacity(capacity){
		islockfree = capacity == 0 || _data.front().is_lock_free();
	}
	~LockFreeArrayQueue() {}

	bool is_lock_free() {
		return islockfree;
	}

	bool isFull() { return _size.load() == _capacity; }
	bool isEmpty() { return _size.load() == 0; }

	bool push(T val, Strategy strategy = Strategy::FORCE);
	bool pop(T& val, Strategy strategy = Strategy::FORCE);

private:
	const int Empty = 0;  //元素为0时，表示为空
	const int Exclude = -1;  //front or rear为-1时，表示其他线程已加锁，正在操作数据

	const int _capacity;  //数组最大容量
	std::vector<atomic<T>>_data;
	std::atomic<int>_size = { 0 };  //当前size
	std::atomic<int>_front = { 0 };  //头指针
	std::atomic<int>_rear = { 0 };   //尾指针
	bool islockfree;
};

template<typename T>
bool LockFreeArrayQueue<T>::push(T val, Strategy strategy) {
	int rear = _rear.load();
	while (true) {
		if (rear == Exclude || isFull()) {
			switch (strategy) {
			case Strategy::YIELD:
				std::this_thread::yield();
			case Strategy::FORCE:
				rear = _rear.load();
				continue;
			}
			return false;
		}
		//加rear锁
		if (_rear.compare_exchange_weak(rear, Exclude)) {
			//已满，失败解锁回退
			if (isFull()) {
				int excepted = Exclude;
				bool flag = _rear.compare_exchange_weak(excepted, rear);
				assert(flag);
				continue;
			}
			break;
		}
	}
	_data[rear].store(val);
	++_size; //必须在解锁前面
	int excepted = Exclude;
	//释放rear锁
	bool flag = _rear.compare_exchange_weak(excepted, (rear + 1) % _capacity);
	assert(flag);
	return true;
}

template<typename T>
bool LockFreeArrayQueue<T>::pop(T& val, Strategy strategy) {
	int front = _front.load();
	while (true) {
		if (front == Exclude || isEmpty()) {
			switch (strategy) {
			case Strategy::YIELD:
				std::this_thread::yield();
			case Strategy::FORCE:
				front = _front.load();
				continue;
			}
			return false;
		}
		//加锁
		if (_front.compare_exchange_weak(front, Exclude)) {
			//空，失败解锁回退
			if (isEmpty()) {
				int excepted = Exclude;
				bool flag = _front.compare_exchange_weak(excepted, front);
				assert(flag);
				continue;
			}
			break;
		}
	}
	val = _data[front].load();
	//_data[front].store(Empty);
	--_size; //必须在解锁前面
	int excepted = Exclude;
	bool flag = _front.compare_exchange_weak(excepted, (front + 1) % _capacity);
	assert(flag);
	return true;
}


std::mutex mx;
condition_variable cond;
bool running = false;
atomic<int>cnt {0};

//队列
LockFreeArrayQueue<int> queue(1000);

const int pn = 100, cn = 100;  //生产者/消费者线程数
//每个生产者push batch*num的数据
const int batch = 100;  
const int num = 100;
unordered_map<int, int> counts[cn];

//生产者
void produce() {
	{
		std::unique_lock<std::mutex>lock(mx);
		cond.wait(lock, []() {return running; });
	}
	for (int i = 0; i < batch; ++i) {
		for (int j = 1; j <= num; ++j)
			queue.push(j);
	}
	++cnt;
}

//消费者
void consume(int i) {
	unordered_map<int, int>& count = counts[i];
	{
		std::unique_lock<std::mutex>lock(mx);
		cond.wait(lock, []() {return running; });
	}
	int val;
	while (true) {
		bool flag = queue.pop(val, LockFreeArrayQueue<int>::Strategy::ABANDON);
		if (flag) ++count[val];
		else if (cnt == pn) break;
	}

}
int main() {
	vector<thread>pThreads, cThreads;
	for (int i = 0; i < pn; ++i)
		pThreads.push_back(thread(&produce));
	for (int i = 0; i < cn; ++i) {
		cThreads.push_back(thread(&consume, i));
	}
    auto start = std::chrono::high_resolution_clock::now();
	{
		std::lock_guard<std::mutex>guard(mx);
		running = true;
		cond.notify_all();
	}
	for (int i = 0; i < pn; ++i) pThreads[i].join();
	for (int i = 0; i < cn; ++i) cThreads[i].join();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

	//结果统计
	unordered_map<int, int> res;
	for (auto& count : counts) {
		for (auto& kv : count) {
			res[kv.first] += kv.second;
		}
	}
	int total = 0;
	int failed = 0;
	for (auto& kv : res) {
		total += kv.second;
		cout << kv.first << " " << kv.second << endl;
		failed = !(kv.first > 0 && kv.first <= num && kv.second == batch * pn);
	}
	cout << "is_lock_free: " << (queue.is_lock_free() ? "true" : "false") << endl;
	cout << "consume time: " << elapsed.count()/1000.0 <<" seconds"<< endl;
	cout << "total numbers: " << total << endl;
	cout << "failed: " << failed << endl;
}
