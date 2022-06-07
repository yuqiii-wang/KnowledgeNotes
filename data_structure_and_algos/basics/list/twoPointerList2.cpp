/*
Given a linked list, return the node where the cycle begins. If there is no cycle, return null.

To represent a cycle in the given linked list, we use an integer pos which represents the position (0-indexed) in the linked list where tail connects to. If pos is -1, then there is no cycle in the linked list.

Note: Do not modify the linked list.
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

    ListNode * detectCycle(ListNode *head);
    ListNode* constructList(int* list, int lsLen, int pos);
    int findPosition(ListNode *posNode);
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

ListNode* Solution::detectCycle(ListNode *head) {
    if (!head || !head->next) return NULL;
    ListNode* fast = head; 
    ListNode* slow = head;
    ListNode* posNode = head;
    // the logic is that when *fast finishes its loop (meeting with *slow), *slow has only finished half of *fast,
    // then when *posNode meets *slow, *slow will have finished same steps as *fast
    while(fast->next && fast->next->next && slow->next){
        fast = fast->next->next;
        slow = slow->next;
        if (fast == slow){
            while(slow != posNode){
                posNode = posNode->next;
                slow = slow->next;
            }
            return posNode;
        }
    }
    return NULL;
}

int Solution::findPosition(ListNode *posNode){
    int idx = 0;
    ListNode* headCopy = head; // to avoid contamination of the list
    while(headCopy){
        if (posNode == headCopy) return idx;
        else idx++;
        headCopy = headCopy->next;
    }
    return idx;
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

    cout << "\n" << SolutionObj->findPosition(SolutionObj->detectCycle(head)) << endl;
    return 0;
}