// You are a professional robber planning to rob houses along a street. 
// Each house has a certain amount of money stashed, 
// the only constraint stopping you from robbing each of them is that,
// adjacent houses have security system connected
// and it will automatically contact the police if two adjacent houses were broken into on the same night.

// Given a list of non-negative integers representing the amount of money of each house, 
// determine the maximum amount of money you can rob tonight without alerting the police.
var max = function(a, b){
    return a > b ? a : b;
}

var rob = function(nums) {
    let optionA = optionB = 0;
    for (let i = 0; i < nums.length; i++){
        if (i % 2 == 0)
            optionA = max(optionA + nums[i], optionB);
        else
            optionB = max(optionB + nums[i], optionA);
    }
    return max(optionA, optionB);
};

inputArr = [1,3,1,1,5];
console.log(rob(inputArr));
inputArr = [1,2,3,1];
console.log(rob(inputArr));
inputArr = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
console.log(rob(inputArr));
inputArr = [2,3,2];
console.log(rob(inputArr));
inputArr = [1,2,1,1];
console.log(rob(inputArr));