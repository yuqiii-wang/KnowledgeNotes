// https://leetcode.com/problems/find-median-from-data-stream/

// insert int to an array-like data structure
// find a median given an array of int

#include <queue>
#include <iostream>

using namespace std;

/** use priority_queue to handle element insertion */
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int> q_small;
    priority_queue<int> q_large;
    
    MedianFinder() {
    }
    
    void addNum(int num) {
        q_small.push(num); // O(log n) insertion
        q_large.push(-q_small.top());   // get the biggest val from q_small
                                        // negative sign added so that the highest val in
                                        // q_small would appear at the bottom on q_large
        q_small.pop(); // remove the highest val
        if (q_small.size() < q_large.size()){   // to maintain the sizes of two are
                                                // half  of the total number added
            q_small.push(-q_large.top());
            q_large.pop();
        }
    }
    
    double findMedian() {
        return (q_small.size() < q_large.size())?
        static_cast<double>(q_small.top()) :
        static_cast<double>(q_small.top() - q_large.top()) / 2.0;
        
    }
};

int main(){
    MedianFinder* M = new MedianFinder();
    M->addNum(2);
    M->addNum(4);
    M->addNum(6);
    M->addNum(1);
    M->addNum(0);
    
    cout << M->findMedian() << endl;
    
    return 0;
}
