#include <unordered_map>
#include <iostream>
#include <list>

/*
    You are given two non-empty linked lists representing two non-negative integers. 
    The digits are stored in reverse order, and each of their nodes contains a single digit. 
    Add the two numbers and return the sum as a linked list.
    
    Constraints:
        0 <= Node.val <= 9
        It is guaranteed that the list represents a number that does not have leading zeros.

    Input: l1 = [2,4,3], l2 = [5,6,4]
    Output: [7,0,8]
    Explanation: 342 + 465 = 807.

    Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
    Output: [8,9,9,9,0,0,0,1]
*/

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {

        ListNode* numListNode = new ListNode();
        ListNode* root = numListNode;
        int carry = 0;
        while (l1 || l2) {
            if (!l1) {
                l1 = new ListNode(0);
            }
            if (!l2) {
                l2 = new ListNode(0);
            }

            numListNode->val = l1->val + l2->val + carry;
            if (numListNode->val > 9) {
                numListNode->val = numListNode->val - 10;
                carry = 1;
            }
            else {
                carry = 0;
            }
            
            l1 = l1->next;
            l2 = l2->next;
            if (l1 || l2) {
                numListNode->next = new ListNode();
                numListNode = numListNode->next;
            }
        }
        if (carry == 1)
            numListNode->next = new ListNode(1);
        return root;
    }
};