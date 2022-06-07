// https://leetcode.com/explore/learn/card/trie/149/practical-application-ii/1056/

#include <iostream>
#include <vector>

using namespace std;

const int ALPHABET_SIZE = 26;

class trieNode{
public:
    bool isEnd;
    trieNode* children[ALPHABET_SIZE];
    trieNode();
};

trieNode::trieNode(){
    for (int i = 0; i < ALPHABET_SIZE; i++)
        children[i] = NULL;
    isEnd = false;
}

class Solution {
private:
    void backTracking(int now_x, int now_y, trieNode* char_iter, trieNode* tracebackRoot);
    void makeTrie(vector<string>& words);
    bool isSafe(int x, int y);
    bool isValid(int x, int y);
    bool searchTrieWord(string word);
public:
    trieNode* root;
    trieNode* tracebackRoot;
    vector<vector<char>> board;
    vector<string> words;
    vector<string> foundWords;
    Solution();
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words);
};

Solution::Solution(){
    ; // empty constructor
}

void Solution::makeTrie(vector<string>& words){
    root = new trieNode();
    vector<string>::iterator iter;
    for (iter = words.begin(); iter != words.end(); iter++){
        string word = *iter;
        trieNode* char_iter = root;
        for(int i = 0; i < word.length(); i++){
            if (!char_iter->children[word[i] - 'a']){
                char_iter->children[word[i] - 'a'] = new trieNode();
                char_iter = char_iter->children[word[i] - 'a'];
            } else {
                char_iter = char_iter->children[word[i] - 'a'];
            }
        }
        char_iter->isEnd = true;
    }
}

void Solution::backTracking(int x, int y, trieNode* char_iter, trieNode* tracebackRoot){
    char c = board[x][y];
    
    if (char_iter->isEnd)
        return;
    if (char_iter->children[c - 'a'] == NULL)
        return;
    
    tracebackRoot->children[c - 'a'] = new trieNode();
    board[x][y] = '$'; // use '$' to indicate traversed cells
    if (isSafe(x+1, y) && isValid(x+1, y))
        backTracking(x+1, y, char_iter->children[c - 'a'], tracebackRoot->children[c - 'a']);
    if (isSafe(x, y+1) && isValid(x, y+1))
        backTracking(x, y+1, char_iter->children[c - 'a'], tracebackRoot->children[c - 'a']);
    if (isSafe(x-1, y) && isValid(x-1, y))
        backTracking(x-1, y, char_iter->children[c - 'a'], tracebackRoot->children[c - 'a']);
    if (isSafe(x, y-1) && isValid(x, y-1))
        backTracking(x, y-1, char_iter->children[c - 'a'], tracebackRoot->children[c - 'a']);
    board[x][y] = c;
};

bool Solution::isSafe(int x, int y){
    if(x >= 0 && y >= 0 && x < board.size() && y < board[0].size())
        return true;
    return false;
}

bool Solution::isValid(int x, int y){
    if (board[x][y] == '$')
        return false;
    return true;
}

vector<string> Solution::findWords(vector<vector<char>>& board_input, vector<string>& words_input){
    board = board_input;
    words = words_input;
    tracebackRoot = new trieNode();
    makeTrie(words); // build Trie
    for (int i = 0; i < board.size(); i++){
        for (int j = 0; j < board[0].size(); j++){
            backTracking(i, j, root, tracebackRoot);
        }
    }
    for (int i = 0; i < words_input.size(); i++){
        if (searchTrieWord(words_input[i]))
            foundWords.push_back(words_input[i]);
    }
    return foundWords;
};

bool Solution::searchTrieWord(string word){
    trieNode* p = tracebackRoot;
    for (int i = 0; i < word.length(); i++){
        if (p->children[word[i] - 'a'] == NULL)
            return false;
        else
            p = p->children[word[i] - 'a'];
    }
    return true;
}

int main(){
    vector<vector<char>> board = {{'a', 'e', 'i', 'p'}, {'q', 'w', 't', 'p'}, {'p', 'e', 'u', 'l'}, {'g', 'o', 'o', 'd'}};
    vector<string> words = {"quit", "good"};
    
    Solution* S = new Solution();
    vector<string> foundWords = S->findWords(board, words);
    for (int i = 0; i < foundWords.size(); i++)
        cout << foundWords[i] << endl;
    return 0;
}
