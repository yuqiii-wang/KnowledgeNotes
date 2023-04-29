// Make LURCache: 
// prepare a hash map locating elements in a doubly linked list which stores the actual data

#include <unordered_map>
#include <list>

struct DLinkedNode {
    int key, value;
    DLinkedNode* prev;
    DLinkedNode* next;
    DLinkedNode(): key(0), value(0), prev(nullptr), next(nullptr) {}
    DLinkedNode(int _key, int _value): key(_key), value(_value), prev(nullptr), next(nullptr) {}
};

class LRUCache {
private:
    std::unordered_map<int, DLinkedNode*> cache;
    DLinkedNode* head;
    DLinkedNode* tail;
    int size;
    int capacity;

    // new head->next points to tail
    // new head->prev points to the second last node (head)
    // old head becomes the second last node
    // tail->prev should be the new head (node)
    void addToHead (DLinkedNode* node) {
        node->prev = head;
        node->next = head->next; // head->next points to tail by this step
        head->next->prev = node;
        head->next = node;
    }

    void removeNode(DLinkedNode* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev; 
    }

    void moveToHead(DLinkedNode* node) {
        removeNode(node);
        addToHead(node);
    }

    DLinkedNode* getTailThenRemoveTail() {
        DLinkedNode* node = tail->prev;
        removeNode(node);

        return node;
    }

public:
    LRUCache(int _capacity): capacity(_capacity), size(0) {
        head = new DLinkedNode();
        tail = new DLinkedNode();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if ( cache.find(key) == cache.end()) {
            return -1;
        }

        DLinkedNode* node = cache[key];
        moveToHead(node);
        return node->value;
    }

    void put(int key, int value) {
        if ( cache.find(key) == cache.end()) {
            DLinkedNode* node = new DLinkedNode(key, value);
            cache[key] = node;
            addToHead(node);
            size++;
            if (size > capacity) {
                DLinkedNode* tailNode = getTailThenRemoveTail();
                cache.erase(tailNode->key);
                delete tailNode;
            }
        }
        else {
            DLinkedNode* node = cache[key];
            node->value = value;
            moveToHead(node);
        }
    }
};