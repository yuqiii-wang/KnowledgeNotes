// Given a linked list, remove the n-th node from the end of list and return its head.

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

    ListNode * removeEndNthNode(int pos);
    ListNode * constructList(int* list, int lsLen);
};

ListNode* Solution::constructList(int* list, int lsLen){

    ListNode *tmp;
    head = NULL;
    for (int i = lsLen-1; i >= 0; i--){
        tmp = new ListNode(list[i]);
        tmp->next = head;
        head = tmp;
    }

    return head;
}

ListNode* Solution::removeEndNthNode(int n){

    // first reach the top n-th node then *slow starts moving until *fast reaches the end
    ListNode** t1 = &head, *t2 = head;
    for(int i = 1; i < n; ++i)
    {
        t2 = t2->next;
    }
    while(t2->next)
    {
        t1 = &((*t1)->next); // here it avoids replacing *head
        t2 = t2->next;
    }
    *t1 = (*t1)->next;
    return head;
}

int main(){

    int pos = 3;
    const int lsLen = 3;
    int list[lsLen] = {3,2,1};

    Solution* SolutionObj = new Solution();
    SolutionObj->constructList(list, lsLen);
    ListNode* head = SolutionObj->removeEndNthNode(pos);

    for(int i = 0; i < lsLen-1; i++){
        cout << head->val << endl;
        head = head->next;
    }

    return 0;
}