#include<iostream>
#include<atomic>
#include<thread>
#include<assert.h>

#include<vector>
#include<unordered_map>
#include<mutex>
#include<condition_variable>
#include<thread>
#include<iostream>
#include <ctime>

using namespace std;


//保证T应当是trival
//基于链表的无界无锁队列
template<typename T>
class LockFreeLinkedQueue {
public:
	//保证初始化在单线程下完成
	LockFreeLinkedQueue() {
		Node* node = new Node(Empty);
		head.store(node);
		tail.store(node);
		islockfree = node->val.is_lock_free();
	}
	~LockFreeLinkedQueue() {
		T val = Empty;
		while (tryPop(val));
		Node* node = head.load();
		if (node != NULL)
			delete node;
	}
	bool is_lock_free() {
		return islockfree;
	}

	bool isEmpty() { return count.load() == 0; }
	bool isFull() { return false; }

	//push操作，CAS加tail锁
	bool push(T val);

	//pop操作，CAS加head锁
	bool tryPop(T& val);

	//不建议使用，当队列中无元素时，会自旋
	T pop();

private:
	struct Node {
		std::atomic<T> val;
		std::atomic<Node*>next{NULL};
		Node(T val) :val(val) {

		}
	};
	const T Empty = 0;
	std::atomic<int>count = { 0 };  //计数器
	std::atomic<Node*>head;  //头结点
	std::atomic<Node*>tail;   //尾结点
	bool islockfree;
};


//push操作，CAS加tail锁
template<typename T>
bool LockFreeLinkedQueue<T>::push(T val) {
	Node* t = NULL;
	Node* node = new Node(val);
	while (true) {
		//t==NULL，表示tail锁被抢
		if (t == NULL) {
			t = tail.load();
			continue;
		}
		//尝试加tail锁
		if (!tail.compare_exchange_weak(t, NULL))
			continue;
		break;
	}
	t->next.store(node);
	++count;
	Node* expected = NULL;
	//释放tail锁
	bool flag = tail.compare_exchange_weak(expected, t->next);
	assert(flag);
	return true;
}

//pop操作，CAS加head锁
template<typename T>
bool LockFreeLinkedQueue<T>::tryPop(T& val) {
	Node* h = NULL, * h_next = NULL;
	while (true) {
		//h==NULL，表示head锁被抢
		if (h == NULL) {
			h = head.load();
			continue;
		}
		//尝试加head锁
		if (!head.compare_exchange_weak(h, NULL))
			continue;
		h_next = h->next.load();
		//h->next != NULL 且 count == 0  
		//   此时在push函数中数据以及count计数器没有来得及更新，因此进行自旋
		if (h_next != NULL) {
			while (count.load() == 0)
				std::this_thread::yield();  //???
		}
		break;
	}
	Node* expected = NULL;
	Node* desired = h;
	//当h_next==NULL时
	//   表示当前链表为空
	if (h_next != NULL) {
		val = h_next->val;
		delete h;
		desired = h_next;
		--count;
	}
	//CAS head，释放head锁
	bool flag = head.compare_exchange_weak(expected, desired);
	assert(flag);
	return h_next != NULL;
}

//不建议使用，当队列中无元素时，会自旋
template<typename T>
T LockFreeLinkedQueue<T>::pop() {
	Node* h = NULL, * h_next = NULL;
	while (true) {
		//h==NULL，表示head锁被抢
		if (h == NULL) {
			h = head.load();
			continue;
		}
		//尝试加head锁
		if (!head.compare_exchange_weak(h, NULL))
			continue;
		h_next = h->next.load();
		//h->next == NULL 
		//   当前队列为空，是否需要解head锁？
		//h->next != NULL 且 count == 0  
		//   此时在push函数中数据以及count计数器没有来得及更新，因此进行自旋
		while (h_next == NULL || count.load() == 0) {
			std::this_thread::yield();  //???
			if (h_next == NULL)
				h_next = h->next.load();
		}
		break;
	}
	T val = h_next->val;
	delete h;
	--count;
	Node* expected = NULL;
	Node* desired = h_next;
	//CAS head，释放head锁
	bool flag = head.compare_exchange_weak(expected, desired);
	assert(flag);
	return val;
}

std::mutex mx;
condition_variable cond;
bool running = false;
atomic<int>cnt{0};

//队列
LockFreeLinkedQueue<int> queue;

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
		bool flag = queue.tryPop(val);
		if (flag) ++count[val];
		else if (cnt == pn) break;
	}

}
int main() {
    cnt = 0;
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
	cout << "consume time: " << elapsed.count()/1000.0 << " seconds" << endl;
	cout << "total numbers: " << total << endl;
	cout << "failed: " << failed << endl;
}
