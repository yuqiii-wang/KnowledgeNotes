/*
Given a linked list, determine if it has a cycle in it.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Example 1:

Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where tail connects to the second node.
*/

/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
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

    int length;
    ListNode* head;

    bool hasCycle(ListNode *head);
    ListNode* constructList(int* list, int lsLen, int pos);
};

ListNode* Solution::constructList(int* list, int lsLen, int pos){

    ListNode* tmp;
    length = lsLen;
    for (int i = lsLen-1; i >= 0; i--){
        tmp = new ListNode(list[i]);
        tmp->next = head;
        head = tmp;
    }
    // pos == 1 indicates a loop
    if(pos != -1){
        ListNode* endNode = head;
        ListNode* posNode;
        for(int i = 0; i < lsLen-1; i++){
            if (i == pos) posNode = endNode;
            endNode = endNode->next; 
        } // endNode will be nullprt at the end of the loop
        endNode->next = posNode;
    }

    return head;
}

bool Solution::hasCycle(ListNode *head) {
    ListNode* fast = head;
    while(head){
        head = head->next;
        if(fast->next && fast->next->next) fast = fast->next->next;
        else return false;
        if (fast == head) return true;
    }
    return false;
}

int main(){

    int pos = 1;
    const int lsLen = 4;
    int list[lsLen] = {3,2,0,-4};

    Solution* SolutionObj = new Solution();
    ListNode* head = SolutionObj->constructList(list, lsLen, pos);
    ListNode* tmp = head;
    for (int i = 0; i < 10; i++) {
        cout << tmp->val << endl;
        tmp = tmp->next;
    } // if no seg fault there is a loop created

    cout << "\n" << SolutionObj->hasCycle(head) << endl; // 0 for false and 1 for true

    return 0;
}