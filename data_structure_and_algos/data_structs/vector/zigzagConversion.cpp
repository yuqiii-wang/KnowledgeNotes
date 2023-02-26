#include <iostream>
#include <vector>
/*
    The string "PAYPALISHIRING" is written in a transposed zigzag pattern on a given number of rows like this:

    Input: s = "PAYPALISHIRING", numRows = 4
    Output: "PINALSIGYAHRPI"
    Explanation:
    P     I    N
    A   L S  I G
    Y A   H R
    P     I

*/

using namespace std;

/*
    Just use a 2d matrix to store all elements including many empty cells

    Each zigzag pattern should have `numRows+numRows-2` cells then it repeats;
    Each zigzag pattern should have numRows-1 cols, then it continues
*/
class SolutionFullMatrix {
public:
    static string convert(string s, int numRows) {

        if (s.size() < 3 || numRows == 1)
            return s;

        vector<vector<char>> mat;

        int idxPatterns = -1;
        for (int i = 0; i < s.size(); i++) {
            int idxInPattern =  i % (numRows*2-2);
            if (idxInPattern == 0) {
                mat.emplace_back(vector<char>(numRows, '\0'));
                idxPatterns++;
            }
            if (idxInPattern < numRows) {
                mat[idxPatterns][idxInPattern] = s[i];
            }
            else {
                idxPatterns++;
                mat.emplace_back(vector<char>(numRows, '\0'));
                mat[idxPatterns][numRows*2-idxInPattern-2] = s[i];
            }
        }

        string retS;
        retS.resize(s.size());
        int idx = 0;
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < mat.size(); j++) {
                if (mat[j][i] != '\0') {
                    retS[idx] = mat[j][i];
                    idx++;
                }
            }
        }

        return retS;
    }
};

/*
    Zigzag has strong patterns that
    * pattern start char at every (numRows*2-1)*idxPattern
    * the following rows should read and reverse read chars in every pattern
*/
class SolutionByPattern 
{
    public:
    static string convert(string s, int numRows) {

        if (s.size() < 3 || numRows == 1)
            return s;

        string retS;
        retS.resize(s.size()+10);

        int remainCharNums = s.size()%(numRows*2-2);

        int idxRetS = 0;
        int patternNums = s.size()/(numRows*2-2)  ;
        for (int i = 0; i < patternNums; i++) {
            retS[idxRetS++] = s[(numRows*2-2)*i];
        }
        if (remainCharNums > 0) {
            retS[idxRetS++] = s[(numRows*2-2)*patternNums];
        }

        for (int j = 1; j < numRows; j++) {
            for (int i = 0; i < patternNums; i++) {
                if (j == numRows-1) {
                    retS[idxRetS++] = s[(numRows*2-2)*i+j];
                }
                else {
                    retS[idxRetS++] = s[(numRows*2-2)*i+j];
                    retS[idxRetS++] = s[(numRows*2-2)*(i+1)-j];
                }
            }
            if (remainCharNums == 0) {
                continue;
            }
            else // remaining chars
            {
                int i = patternNums;

                if ( (numRows*2-2)*i+j < s.size() && idxRetS < s.size() ) {
                    retS[idxRetS++] = s[(numRows*2-2)*i+j];
                }
                if ( (numRows*2-2)*(i+1)-j < s.size() && idxRetS < s.size() ) {
                    retS[idxRetS++] = s[(numRows*2-2)*(i+1)-j];
                }
                if (j == numRows-1 && (numRows*2-2)*i+j < s.size() && idxRetS < s.size() ) {
                    retS[idxRetS++] = s[(numRows*2-2)*i+j];
                }
            }
        }
        return retS;
    }
};

int main() {

    string s{"PAYPALISHIRING"};

    cout << SolutionFullMatrix::convert(s, 4) << endl;
    cout << SolutionByPattern::convert(s, 4) << endl;

    string s2{"ABC"};
    cout << SolutionFullMatrix::convert(s2, 3) << endl;
    cout << SolutionByPattern::convert(s2, 3) << endl;

    string s3{"ABCDE"};
    cout << SolutionFullMatrix::convert(s3, 4) << endl;
    cout << SolutionByPattern::convert(s3, 4) << endl;

    string s4{"Apalindromeisaword,phrase,number,orothersequenceofunitsthatcanbereadthesamewayineitherdirection,withgeneralallowancesforadjustmentstopunctuationandworddividers."};
    cout << SolutionFullMatrix::convert(s4, 4) << endl;
    cout << SolutionByPattern::convert(s4, 4) << endl;

    cout << SolutionFullMatrix::convert(s, 3) << endl;
    cout << SolutionByPattern::convert(s, 3) << endl;

    return 0;
}