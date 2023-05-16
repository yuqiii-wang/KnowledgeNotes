// Given a vector of strings, find the count of a matching word

#include <vector>
#include <iostream>

class Solution {
public:
    static int findArticleWordCount(std::vector<std::string>& article, std::string word)
    {
        int count = 0;
        for (auto& sentence : article) {
            auto idx = sentence.find(word);
            while (idx != std::string::npos) {
                count++;
                for (int i = idx; i < idx + word.size(); i++) {
                    sentence[i] = '0';
                }
                idx = sentence.find(word);
            }
        }
        return count;
    }
};

int main()
{
    std::vector<std::string> article{"ssssll", "aaaa", "bbb", "ccllcc"};
    std::string word("ll");
    std::cout << Solution::findArticleWordCount(article, word) << std::endl;
    return 0;
}