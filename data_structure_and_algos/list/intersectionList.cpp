// Write a program to find the node at which the intersection of two singly linked lists begins.

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

    ListNode *head, *headA, *headB;

    ListNode * detectIntersection(ListNode *headA, ListNode *headB);
    ListNode* constructSubList(int* list, int lsLen);
    ListNode* constructList(int* listA, int lsLenA, int* listB, int lsLenB, int pos);
    int findPosition(ListNode *posNode);
};

ListNode* Solution::constructList(int* listA, int lsLenA, int* listB, int lsLenB, int pos){
    // assume list A is the main list and list B is attched to A
    // if pos == -1, there is no intersection
    headA = constructSubList(listA, lsLenA);
    headB = constructSubList(listB, lsLenB);

    ListNode *tmpPos = headA, *tmpEndB = headB;
    for (int i = 0; i < pos; i++){
        tmpPos = tmpPos->next;
    }
    for (int i = 0; i < lsLenB-1; i++){
        tmpEndB = tmpEndB->next;
    }
    tmpEndB->next = tmpPos;

    return tmpPos;
}

ListNode* Solution::constructSubList(int* list, int lsLen){

    ListNode *tmp;
    head = NULL;
    for (int i = lsLen-1; i >= 0; i--){
        tmp = new ListNode(list[i]);
        tmp->next = head;
        head = tmp;
    }

    return head;
}

ListNode * Solution::detectIntersection(ListNode *headA, ListNode *headB){
    ListNode *p1 = headA;
    ListNode *p2 = headB;
        
    if (!p1 || !p2) return NULL;

    while (p1 && p2 && p1 != p2) {
        p1 = p1->next;
        p2 = p2->next;

        if (p1 == p2) return p1; // p1 and p2 are same

        // after switch now p1 and p2 are equidistant
        if (!p1) p1 = headB;
        if (!p2) p2 = headA;
    }
        
    return p1;
}

int Solution::findPosition(ListNode *posNode){
    int idx = 0;
    ListNode* headCopy = headA; // to avoid contamination of the list
    while(headCopy){
        if (posNode == headCopy) return idx;
        else idx++;
        headCopy = headCopy->next;
    }
    return idx;
}

int main(){

    int pos = 1;
    const int lsLenA = 4;
    int listA[lsLenA] = {3,2,0,-4};
    const int lsLenB = 2;
    int listB[lsLenB] = {9,8};

    Solution* SolutionObj = new Solution();
    ListNode* intersection = SolutionObj->constructList(listA, lsLenA, listB, lsLenB, pos);
    ListNode* headA = SolutionObj->headA;
    ListNode* headB = SolutionObj->headB;

    // use for loop to prove list is created successfully
    ListNode* tmpNode = headB;
    for(int i = 0; i < 5; i++){
        tmpNode = tmpNode->next;
    }

    ListNode* intersectionDetected = SolutionObj->detectIntersection(headA, headB);
    if (intersectionDetected == intersection) cout << "SUCCESS" << endl;
    else cout << "FAILED" << endl;

    return 0;
}