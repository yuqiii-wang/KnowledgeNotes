/*
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
*/

#include <iostream>

#ifndef NULL
#define NULL nullptr // -std=c++11
#endif

using namespace std;

struct ListNode {
    int val;
    ListNode *next;
    ListNode() = default;
    ListNode(int x) : val(x), next(NULL) {};
};

class Solution {

    public:

    ListNode *head;

    ListNode * reverseList();
    ListNode * constructList(int* list, int lsLen);
};

ListNode* Solution::constructList(int* list, int lsLen){
    ListNode* tmp;
    head = NULL;
    for (int i = lsLen-1; i >= 0; i--){
        tmp = new ListNode(list[i]);
        tmp->next = head;
        head = tmp;
    }
    return head;
}

ListNode* Solution::reverseList(){
    // the principle is to reverse nodes' pointer one by one
    if (!head) 
    return NULL;

    ListNode* counter = head;
    // only loop lsLen-1 times so need head->next
    while(head->next){
        ListNode* tmp = head->next;
        head->next = tmp->next;
        tmp->next = counter; // here the pointer points back to its previous node
        counter = tmp;
    }
    return counter;
}

int main(){

    const int lsLen = 3;
    int list[lsLen] = {3,2,1};

    Solution* SolutionObj = new Solution();
    SolutionObj->constructList(list, lsLen);
    ListNode* head = SolutionObj->reverseList();
    for(int i = 0; i < lsLen; i++){
        cout << head->val << "\t";
        head = head->next;
    }

    return 0;
}