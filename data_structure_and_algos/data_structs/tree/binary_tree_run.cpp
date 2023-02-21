#include "binary_tree_insert.hpp"
#include "binary_tree_search.hpp"


int main() 
{
    Node* root  = new Node(11);
    Node* nodeList[10];
    for (int i = 0; i < 10; i++)
        nodeList[i] =  new Node(i*2);

    for (int i = 0; i < 10; i++)
        insertBinaryTreeNode(root, nodeList[i]);

    std::cout << "========printBinaryTreeInOrder========" << std::endl;
    printBinaryTreeInOrder(root);

    std::cout << "========searchBinaryTreeNode========" << std::endl;
    for (int i = 0; i < 10; i++) {
        Node* findNode = searchBinaryTreeNode(root, i);
        if (findNode) {
            std::cout << "found: " << findNode->data << ", ";
        }
        else {
            std::cout << "not found: " << i << ", ";
        }
    }
    std::cout << std::endl;

    std::cout << "========searchBinaryTreeNode========" << std::endl;


    for (int i = 0; i < 10; i++)
        delete nodeList[i];
    delete root;

    return 0;
}