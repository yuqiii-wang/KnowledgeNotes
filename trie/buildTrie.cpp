// https://leetcode.com/explore/learn/card/trie/147/basic-operations/1047/
// https://www.geeksforgeeks.org/trie-insert-and-search/

#include <iostream>

using namespace std;

const int ALPHABET_SIZE = 26;

// Trie is an efficient tool for string manipulation
class trieNode{
public:
    trieNode* children[ALPHABET_SIZE];
    bool isEnd;
    trieNode();
};

trieNode::trieNode(){
    for (int i = 0; i < ALPHABET_SIZE; i++)
        children[i] = NULL;
    isEnd = false;
}

class buildTrie{
    struct trieNode* root;
    
public:
    buildTrie();
    bool insert(string str);
    bool search(string str);
};

buildTrie::buildTrie(){
    root = new trieNode();
}

bool buildTrie::insert(string str){
    trieNode *p = root;
    for (int i = 0; i < str.length(); i++){
        if (p->children[str[i] - 'a'] == NULL){
            p->children[str[i] - 'a'] = new trieNode();
            p = p->children[str[i] - 'a'];
        }
    }
    p->isEnd = true;
    return true;
}

bool buildTrie::search(string str){
    trieNode *p = root;
    for (int i = 0; i < str.length(); i++){
        if (p->children[str[i] - 'a'] == NULL){
            return false;
        }
        else
            p = p->children[str[i] - 'a'];
    }
    if (p->isEnd)
        return true;
    else
        return false;
}

int main(){
    buildTrie* Trie = new buildTrie();
    Trie->insert("hello");
    cout << Trie->search("hello") << endl;
    cout << Trie->search("world") << endl;
    return 0;
}
