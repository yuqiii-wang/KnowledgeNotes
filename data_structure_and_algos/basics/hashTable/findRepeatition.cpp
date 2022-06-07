// https://leetcode.com/problems/repeated-dna-sequences/

// use hash to store sub-str to achieve O(n)

#include <iostream>
#include <vector>
#include <unordered_map>

using namespace std;

class Solution {
public:
    vector<string> findRepeatedDnaSequences(string s);
};

vector<string> Solution::findRepeatedDnaSequences(string s){
    unordered_map<string, int> prevSeen;
    vector<string> repeat;

    for (int i = 0; i + 9 < s.size(); i++){
        if (prevSeen.find(s.substr (i, 10)) == prevSeen.end()){ // when first seen
            prevSeen.insert({s.substr(i, 10), 1});
        } else if (prevSeen[s.substr (i, 10)] == 1) {           // when seen for the second time
            repeat.push_back(s.substr(i, 10));
            prevSeen[s.substr (i, 10)]++;
        } else {                                                // otherwise (seen for more than twice)
            prevSeen[s.substr (i, 10)]++;
        }
    }
    return repeat;
}

int main(int argc, const char * argv[]) {
    
    Solution* S = new Solution();
    
    vector<string> repeat = S->findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT");
    for (int i = 0; i < repeat.size(); i++)
        cout << repeat[i] << ' ';
    cout << endl;
    
    vector<string> repeat2 = S->findRepeatedDnaSequences("AAAAAAAAAAAA");
    for (int i = 0; i < repeat2.size(); i++)
        cout << repeat2[i] << ' ';
    cout << endl;
    
    vector<string> repeat3 = S->findRepeatedDnaSequences("AAAAAAAAAAA");
    for (int i = 0; i < repeat3.size(); i++)
        cout << repeat3[i] << ' ';
    cout << endl;
    
    vector<string> repeat4 = S->findRepeatedDnaSequences("AAAAAAAAAA");
    for (int i = 0; i < repeat4.size(); i++)
        cout << repeat4[i] << ' ';
    cout << endl;
    
    return 0;
}
