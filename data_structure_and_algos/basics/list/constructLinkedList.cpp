/*
Design your implementation of the linked list. You can choose to use the singly linked list or the doubly linked list. A node in a singly linked list should have two attributes: val and next. val is the value of the current node, and next is a pointer/reference to the next node. If you want to use the doubly linked list, you will need one more attribute prev to indicate the previous node in the linked list. Assume all nodes in the linked list are 0-indexed.

Implement these functions in your linked list class:

get(index) : Get the value of the index-th node in the linked list. If the index is invalid, return -1.
addAtHead(val) : Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
addAtTail(val) : Append a node of value val to the last element of the linked list.
addAtIndex(index, val) : Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
deleteAtIndex(index) : Delete the index-th node in the linked list, if the index is valid.
*/

//compile by g++ -std=gnu++0x -o test constructLinkedList.cpp

#include <iostream>

using namespace std;

class cLinkedCls{
    private:
        struct node{
            int val;
            node* next;
            node() = default;
            node(int val, node* next = nullptr):val(val), next(next){};
        };
        int length;
        node* head;
    public:
        cLinkedCls() = default;
        cLinkedCls(node* firstNode): length(1), head(firstNode){};

        node* insertNode(int idx, int val);
        node* getNode(int idx);
        node* addTail(int val);
        node* addHead(int val);
        node* deleteNode(int idx);
        node* deleteHead();
        node* deleteTail();
        node* outputList();
};

cLinkedCls::node* cLinkedCls::outputList(){
    return head;
}

cLinkedCls::node* cLinkedCls::addHead(int val){
    node* new_node = insertNode(0, val);
    return new_node;
}

cLinkedCls::node* cLinkedCls::addTail(int val){
    node* new_node = insertNode(length, val);
    return new_node;
}

cLinkedCls::node* cLinkedCls::insertNode(int idx, int val){
    if (idx <= length && idx > 0){
        node* new_node = new node(val);
        node* tmp = head;
        node* tmp_copy;

        int count = idx;
        while(--count > 0) tmp = tmp->next;

        tmp_copy = tmp->next;
        tmp->next = new_node;
        new_node->next = tmp_copy;

        length++;
        return tmp;
    }
    else if (idx == 0){
         node* tmp = new node(val);
         tmp->next = head;
         head = tmp;
         length++;
         return head;
    }
    else return nullptr;
}

cLinkedCls::node* cLinkedCls::getNode(int idx){
    if (idx < length){
        node* tmp = head;

        int count = idx;
        while(count-- > 0) tmp = tmp->next;

        return tmp;
    }
    else return nullptr;
}

cLinkedCls::node* cLinkedCls::deleteHead(){
    return deleteNode(0);
}

cLinkedCls::node* cLinkedCls::deleteTail(){
    return deleteNode(length-1);
}

cLinkedCls::node* cLinkedCls::deleteNode(int idx){
    if (idx < length && idx > 0){
        node* tmp = head;
        node* tmp_copy;

        int count = idx;
        while(count-- > 0) tmp = tmp->next;

        tmp = tmp->next;

        length--;
        return tmp;
    }
    else if (idx == 0){
        node* tmp = head;
        int count = idx;

        while(count-- > 0) tmp = tmp->next;
        head = tmp->next;

        length--;
        return tmp;
    }
    else return nullptr;
}


// int main(){
//     cLinkedCls* nodeList = new cLinkedCls();
//     nodeList->addHead(1);
//     nodeList->addHead(12);
//     nodeList->addHead(123);
//     nodeList->addHead(1234);
//     nodeList->addTail(12345);
//     nodeList->insertNode(0, 123456);
//     for (int i = 0; i < 6; i++){
//         cout << nodeList->getNode(i)->val << endl;
//     }

//     cout << endl;

//     nodeList->deleteNode(0);
//     nodeList->deleteNode(5);
//     nodeList->deleteNode(2);
//     for (int i = 0; i < 3; i++){
//         cout << nodeList->getNode(i)->val << endl;
//     }

//     return 0;
// }