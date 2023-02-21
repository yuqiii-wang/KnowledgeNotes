#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;

/*
    Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

    The overall run time complexity should be O(log (m+n)).

    Example 1:

    Input: nums1 = [1,3], nums2 = [2]
    Output: 2.00000
    Explanation: merged array = [1,2,3] and median is 2.
*/


class SolutionByBuiltinSort {
public:
    static double findMedianSortedArrays(std::vector<int>& nums1, std::vector<int>& nums2) {
        if (nums1.size() == 1 && nums2.size()==0)
            return nums1[0];
        if (nums1.size() == 0 && nums2.size()==1)
            return nums2[0];
        nums1.reserve(nums1.size()+nums2.size());
        nums1.insert(nums1.end(), nums2.begin(), nums2.end());
        std::sort(nums1.begin(), nums1.end());
        int idx = nums1.size() / 2;
        if (nums1.size() % 2 != 0)
            return nums1[idx];
        else
            return (nums1[idx] + nums1[idx-1])/2.0;
    }
};

/*
    Binary search basically means by certain condition split a vector into two halves (not necessarily of equal sizes),
    then discard one half.
    Given the remaining half, continue spliting it into two halves, 
    again discard one half until meeting a termination condition.

    In this case of two sorted arrays, the median index should be 
    `k = (nums1.size() + nums2.size()) / 2` of the combined array `nums1.insert(nums1.end(), nums2.begin(), nums2.end());`.
    * For `nums1[k/2] < nums2[k/2]`, the median does not exist in nums1[0, ... ,k/2], 
    instead should exist between nums1[k/2, ..., nums1Size],  nums2[0, ..., nums2Size], and vice versa.
    If nums1[k/2] or nums2[k/2] has boundary breach, simply set the last element nums1[-1] or nums2[-1] for comparison 
    against the another array's nums1[k/2] or nums2[k/2]
    * Then k update should be
    ```cpp
        int newIndex1 = min(index1 + k / 2 - 1, m - 1); // int m = nums1.size(); for boundary breach checking
        int newIndex2 = min(index2 + k / 2 - 1, n - 1); // int n = nums2.size(); for boundary breach checking
        int pivot1 = nums1[newIndex1];
        int pivot2 = nums2[newIndex2];
        if (pivot1 <= pivot2) {
            k -= newIndex1 - index1 + 1; // index is init to zero in the first k update,
                                         // in the second k update, 
            index1 = newIndex1 + 1;
        }
        else {
            k -= newIndex2 - index2 + 1;
            index2 = newIndex2 + 1;
        }
    ```
*/

class SolutionBinarySearch {
public:
    int getKthElement(const vector<int>& nums1, const vector<int>& nums2, int k) {

        int m = nums1.size();
        int n = nums2.size();
        int index1 = 0, index2 = 0;

        while (true) {
            // edge conditions
            if (index1 == m) {
                return nums2[index2 + k - 1];
            }
            if (index2 == n) {
                return nums1[index1 + k - 1];
            }
            if (k == 1) {
                return min(nums1[index1], nums2[index2]);
            }

            // update k
            int newIndex1 = min(index1 + k / 2 - 1, m - 1);
            int newIndex2 = min(index2 + k / 2 - 1, n - 1);
            int pivot1 = nums1[newIndex1];
            int pivot2 = nums2[newIndex2];
            if (pivot1 <= pivot2) {
                k -= newIndex1 - index1 + 1;
                index1 = newIndex1 + 1;
            }
            else {
                k -= newIndex2 - index2 + 1;
                index2 = newIndex2 + 1;
            }
        }
    }

    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int totalLength = nums1.size() + nums2.size();
        if (totalLength % 2 == 1) {
            return getKthElement(nums1, nums2, (totalLength + 1) / 2);
        }
        else {
            return (getKthElement(nums1, nums2, totalLength / 2) + getKthElement(nums1, nums2, totalLength / 2 + 1)) / 2.0;
        }
    }
};

int main()
{
    std::vector<int> a{0,1,19,20};
    std::vector<int> b{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};

    SolutionByBuiltinSort::findMedianSortedArrays(a,b);

    SolutionBinarySearch solutionBinarySearch;
    std::cout << solutionBinarySearch.findMedianSortedArrays(a,b) << std::endl;

    std::vector<int> c{};
    std::vector<int> d{1,2,3,4,5,6};
    std::cout << solutionBinarySearch.findMedianSortedArrays(c,d) << std::endl;

    std::vector<int> e{1,2,3};
    std::vector<int> f{1,2,3,4,5};
    std::cout << solutionBinarySearch.findMedianSortedArrays(e,f) << std::endl;

    return 0;
}