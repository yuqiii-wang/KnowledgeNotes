struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {

        if (head == nullptr)
            return head;

        int n = 1;
        ListNode* tmp = head;
        while (tmp->next != nullptr) {
            tmp = tmp->next;
            n++;
        }

        tmp->next = head;
        int breakIdx = n - k % n ;
        while (breakIdx--) {
            tmp = tmp->next; 
        }

        ListNode* ret = tmp->next;
        tmp->next = nullptr;

        return ret;
    }
};