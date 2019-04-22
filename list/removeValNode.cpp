/*
Remove all elements from a linked list of integers that have value val.

Example:

Input:  1->2->6->3->4->5->6, val = 6
Output: 1->2->3->4->5
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

    ListNode * removeValNode(int val);
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

ListNode * Solution::removeValNode(int val){
    if (!head)
    return NULL;

    ListNode** tmp =&head;
    while(*tmp){
        if ((*tmp)->val == val)
        *tmp = (*tmp)->next;
        else
        tmp = &((*tmp)->next);

    }
    return head;
}

int main(){

    const int lsLen = 3;
    int list[lsLen] = {3,2,1};

    Solution* SolutionObj = new Solution();
    SolutionObj->constructList(list, lsLen);
    ListNode* head = SolutionObj->removeValNode(2);
    for(int i = 0; i < lsLen-1; i++){
        cout << head->val << endl;
        head = head->next;
    }

    return 0;
}