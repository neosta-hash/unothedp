// unothedp.cpp : Defines the entry point for the console application.
//
#include <vector>
#include <stack>
#include <queue>
#include <iostream>
#include <math.h>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

using namespace std;

#define BP (cout<<endl)

#define DOCK() do{	\
	int dock;	\
	cin>>dock;	\
}while(0)

template<typename T>
void printContainer(vector<T> &container)
{
	for (auto it = container.begin(); it != container.end(); it++)
	{
		cout << *it;
		if (container.end() - 1 != it)
			cout << ", ";
	}
}

class Solution
{
public:
	// 121. Best Time to Buy and Sell Stock
	int maxProfit(vector<int>& prices) {
		int n = prices.size();

		if (n == 0)
			return 0;

		int buy = prices[0];
		int mp = 0;

		for (int i = 1; i < n; ++i) {
			if (prices[i] < buy)
				buy = prices[i];
			else
				mp = max(mp, prices[i] - buy);
		}

		return mp;
	}

	// 122. Best Time to Buy and Sell Stock II
	int maxProfitII(vector<int>& prices) {
		int n = prices.size();

		if (n == 0)
			return 0;

		int mp = 0;

		for (int i = 1; i < n; ++i) {
			// int profit = prices[i] - prices[i - 1];
			// if (profit > 0)
			// 	mp += profit;

			mp += max(0, prices[i] - prices[i - 1]);
		}

		return mp;
	}

	// 309. Best Time to Buy and Sell Stock with Cooldown
	int maxProfitWithCooldown(vector<int>& prices) {
		int n = prices.size();

		if (n <= 1)
			return 0;

		vector<int> buy(n, 0);
		vector<int> sell(n, 0);

		buy[0] = -prices[0];
		buy[1] = max(-prices[0], -prices[1]);
		sell[1] = max(0, buy[0] + prices[1]);

		for (int i = 2; i < n; ++i) {
			buy[i] = max(buy[i - 1], sell[i - 2] - prices[i]);
			sell[i] = max(sell[i - 1], buy[i - 1] + prices[i]);
		}

		return sell[n - 1];
	}

	// 714. Best Time to Buy and Sell Stock with Transaction Fee
	int maxProfitWithTransactionFee(vector<int>& prices, int fee) {
		int n = prices.size();

		if (n <= 1)
			return 0;

		int buy = -prices[0] - fee;
		int sell = 0;

		for (int i = 1; i < n; ++i) {
			int hold = buy;
			buy = max(buy, sell - prices[i] - fee);
			sell = max(sell, hold + prices[i]);
		}

		return sell;
	}

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	int maxProfitAtMostTwoTransactions(vector<int>& prices) {
		if (prices.size() <= 1)
			return 0;

		int phases = 5;
		vector<int> dp1(phases, INT_MIN);
		vector<int> dp2(phases, 0);
		dp1[0] = 0;

		for (auto price : prices) {
			for (int i = 0; i < phases; ++i) {
				dp2[i] = dp1[i];
				if (i % 2 && dp1[i - 1] != INT_MIN)
					dp2[i] = max(dp1[i], dp1[i - 1] - price);
				else if (i > 0 && dp1[i - 1] != INT_MIN)
					dp2[i] = max(dp1[i], dp1[i - 1] + price);
			}

			swap(dp1, dp2);
		}

		int mp = dp1[0];
		for (int i = 2; i < phases; i += 2)
			mp = max(mp, dp1[i]);

		return mp;
	}

	// 188. Best Time to Buy and Sell Stock IV (at Most K Transactions)
	int maxProfitAtMostKTransactions(int k, vector<int>& prices) {
		if (prices.size() <= 1)
			return 0;

		int phases = k * 2 + 1;
		vector<int> dp1(phases, INT_MIN);
		vector<int> dp2(phases, 0);
		dp1[0] = 0;

		for (auto price : prices) {
			for (int i = 0; i < phases; ++i) {
				dp2 [i] = dp1[i];
				if (i % 2 && dp1[i - 1] != INT_MIN)
					dp2[i] = max(dp1[i], dp1[i - 1] - price);
				else if (i > 0 && dp1[i - 1] != INT_MIN)
					dp2[i] = max(dp1[i], dp1[i - 1] + price);
			}

			swap(dp1, dp2);
		}

		int mp = 0;
		for (int i = 2; i < phases; i += 2)
			mp = max(mp, dp1[i]);

		return mp;
	}

	// 70. Climbing Stairs
	int climbStairs(int n) {
		vector<int> dp = {0, 1, 2};

		for (int i = 3; i <= n; ++i)
			dp.emplace_back(dp[i - 1] + dp[i - 2]);

		return dp[n];
	}

	// 746. Min Cost Climbing Stairs
	int minCostClimbingStairs(vector<int>& cost) {
		int n = cost.size();

		if (n == 0)
			return 0;
		else if (n == 1)
			return cost.front();
		else if (n == 2)
			return min(cost[0], cost[1]);

		vector<int> dp = {cost[0], cost[1]};

		for (int i = 2; i < n; ++i)
			dp.emplace_back(min(dp[i - 1], dp[i - 2]) + cost[i]);

		return min(dp[n - 1], dp[n - 2]);
	}

	// 413. Arithmetic Slices
	int numberOfArithmeticSlices(vector<int>& nums) {
		int n = nums.size();

		if (n < 3)
			return 0;

		vector<int> dp(n, 0);
		int num = 0;

		for (int i = 2; i < n; ++i) {
			if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
				dp[i] = dp[i - 1] + 1;
				num += dp[i];
			}
		}

		return num;
	}

	// 300. Longest Increasing Subsequence
	int lengthOfLIS(vector<int>& nums) {
		int n = nums.size();

		if (n == 0)
			return 0;

		vector<int> dp(n, 1);
		int len = 1;

		for (int i = 1; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				if (nums[i] > nums[j])
					dp[i] = max(dp[i], dp[j] + 1);
			}
			len = max(len, dp[i]);
		}

		return len;
	}

	/**
	 * TODO: Follow up
	 * Binary search
	 */
	// int lengthOfLIS(vector<int>& nums) {
	// 	int n = nums.size();
	// }

	// 646. Maximum Length of Pair Chain
	 int findLongestChain(vector<vector<int>>& pairs) {
		int n = pairs.size();

		int len = 

		return len;
	}
};

int main()
{
	Solution solu;

	// 121. Best Time to Buy and Sell Stock
	// vector<int> prices = { 7,1,5,3,6,4 };
	// // prices = { 7,6,4,3,1 };
	// // prices = {};
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfit(prices) << endl << endl;

	// 122. Best Time to Buy and Sell Stock II
	// vector<int> prices = { 7,1,5,3,6,4 };
	// // prices = { 7,6,4,3,1 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitII(prices) << endl << endl;

	// 309. Best Time to Buy and Sell Stock with Cooldown
	// vector<int> prices = { 7,1,5,3,6,4 };
	// prices = { 1,2,3,0,2 };
	// // prices = { 2,1,2,1,0,0,1 };
	// // prices = { 1,2,4,2,5,7,2,4,9,0 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitWithCooldown(prices) << endl << endl;

	// 714. Best Time to Buy and Sell Stock with Transaction Fee
	// vector<int> prices = { 1, 3, 2, 8, 4, 9 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// int fee = -1;
	// while(fee < 0)
	// {
	// 	cout << "Transaction fee: ";
	// 	cin >> fee;
	// }
	// cout << "Max profit: " << solu.maxProfitWithTransactionFee(prices, fee) << endl << endl;

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	// vector<int> prices = { 3,3,5,0,0,3,1,4 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// cout << "Max profit: " << solu.maxProfitAtMostTwoTransactions(prices) << endl << endl;

	// 188. Best Time to Buy and Sell Stock IV (at Most K Transactions)
	// vector<int> prices = { 3,3,5,0,0,3,1,4 };
	// prices = { 2,4,1 };
	// prices = { 3,2,6,5,0,3 };
	// cout << "prices: [ ";
	// printContainer(prices);
	// cout << " ]" << endl;
	// int k = -1;
	// while (k <= 0)
	// {
	// 	cout << "Transacions: ";
	// 	cin >> k;
	// }
	// cout << "Max profit: " << solu.maxProfitAtMostKTransactions(k, prices) << endl << endl;

	// 70. Climbing Stairs
	// int n;
	// while (1)
	// {
	// 	cout << "How many stairs do you wanna climb: ";
	// 	cin >> n;
	// 	cout << "For total " << n << " stairs, distinct ways: " << solu.climbStairs(n) << endl << endl;
	// }

	// 746. Min Cost Climbing Stairs
	// vector<int> cost = { 10, 15, 20 };
	// // cost = { 1, 100, 1, 1, 1, 100, 1, 1, 100, 1 };
	// // cost = { 1, 0, 0, 0 };
	// cout << "Minimum  cost: " << solu.minCostClimbingStairs(cost) << endl << endl;

	// 413. Arithmetic Slices
	// vector<int> A = { 3, -1, -5, -9 };
	// A = { 1,3,5,7,9 };
	// cout << "A: [ ";
	// printContainer(A);
	// cout << " ]" << endl;
	// cout << "Number of arithmetic slices: " << solu.numberOfArithmeticSlices(A) << endl << endl;

	// 300. Longest Increasing Subsequence
	// vector<int> nums = { 10,9,2,5,3,7,101,18 };
	// nums = { 4,10,4,3,8,9 };
	// cout << "nums: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Length of longest increasing subsequence: " << solu.lengthOfLIS(nums) << endl << endl;

	// 646. Maximum Length of Pair Chain
	vector<vector<int>> pairs;
	pairs = {
		{ 1,2 },
		{ 2,3 },
		{ 3,4 }
	};

	pairs = {
		{ 3,4 },
		{ 2,3 },
		{ 1,2 }
	};

	cout << "Length of the longest chain is: " << solu.findLongestChain(pairs) << endl << endl;

	// 376. Wiggle Subsequence
	// vector<int> nums = { 1,7,4,9,2,5 };
	// // nums = { 1,17,5,10,13,15,10,5,16,8 };
	// // nums = { 1,2,3,4,5,6,7,8,9 };
	// cout << "Length of the longest wiggle subsequence is: " << solu.wiggleMaxLength(nums) << endl << endl;

	// 198. House Robber
	// vector<int> nums;
	// nums = { 1,2,3,1 };
	// nums = { 2,7,9,3,1 };
	// nums = { 1,2,3,1,7,6,5,9,2,2,6 };
	// cout << "Money amount list: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Maximum amount of robbery: " << solu.rob(nums) << endl << endl;

	// 213. House Robber II
	// vector<int> nums;
	// nums = { 1,2,3,1 };
	// nums = { 2,7,9,3,1 };
	// //nums = { 1,2,3,1,7,6,5,9,2,2,6 };
	// cout << "Money amount list: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Maximum amount of robbery: " << solu.robII(nums) << endl << endl;

	// 53. Maximum Subarray
	// vector<int> nums = { -2,1,-3,4,-1,2,1,-5,4 };
	// cout << "Money amount list: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Maximum Subarray: " << solu.maxSubArray(nums) << endl << endl;

	// 650. 2 Keys Keyboard
	// while (1)
	// {
	// 	int n = 0;
	// 	while(n <= 0)
	// 	{
	// 		cout << "Number of 'A': ";
	// 		cin >> n;
	// 	}
	// 	cout << "Minimal steps with 2 keys keyboard: " << solu.minStepsWith2KeysKeyboard(n) << endl << endl;
	// }

	// 303. Range Sum Query - Immutable
	// vector<int> nums;
	// nums = { -2, 0, 3, -5, 2, -1 };
	// int i,j; // indices i and j (i <= j)
	// NumArray* na = new NumArray(nums);

	// while (1)
	// {
	// 	cout << "Input index i: ";
	// 	cin >> i;
	// 	while (1)
	// 	{
	// 		cout << "Input index j: ";
	// 		cin >> j;
	// 		if(i <= j)
	// 			break;
	// 	}
	// 	cout << "The sum of elements between indices " << i << " and " << j << ": " << na->sumRange(i, j) << endl << endl;
	// }

	// 304. Range Sum Query 2D - Immutable
	// vector<vector<int>> matrix;
	// matrix = {
	// 	{ 3, 0, 1, 4, 2 },
	// 	{ 5, 6, 3, 2, 1 },
	// 	{ 1, 2, 0, 1, 5 },
	// 	{ 4, 1, 0, 1, 7 },
	// 	{ 1, 0, 3, 0, 5 }
	// };
	// int row1, col1, row2, col2;
	// NumMatrix* nm = new NumMatrix(matrix);

	// while(1)
	// {
	// 	cout << "row1: ";
	// 	cin >> row1;
	// 	cout << "col1: ";
	// 	cin >> col1;
	// 	while (1)
	// 	{
	// 		cout << "row2: ";
	// 		cin >> row2;
	// 		if(row1 <= row2)
	// 			break;
	// 	}
	// 	while (1)
	// 	{
	// 		cout << "col2: ";
	// 		cin >> col2;
	// 		if(col1 <= col2)
	// 			break;
	// 	}
	// 	cout << "Sum of region ((" << row1 << "," << col1 << "),(" << row2 << "," << col2 << ")): " << nm->sumRegion(row1, col1, row2, col2) << endl << endl;
	// }

	// 1218. Longest Arithmetic Subsequence of Given Difference
	// vector<int> arr;
	// arr = { 1,2,3,4 };
	// arr = { 1,5,7,8,5,3,4,2,1 };
	// int difference;

	// while (1)
	// {
	// 	cout << "Difference: ";
	// 	cin >> difference;
	// 	cout << "The lengh of longest arithmetic subsequence for difference " << difference << " is: " << solu.longestSubsequence(arr, difference) << endl << endl;
	// }

	// 392. Is Subsequence
	// string t = "ahbgdc";
	// string s;

	// while (1)
	// {
	// 	cout << "String t: " << t << endl;
	// 	cout << "Inuput string s: ";
	// 	cin >> s;
	// 	cout << "s is a subsequence of t: " << (solu.isSubsequence(s, t) ? "true" : "false") << endl << endl;
	// }

	// 1143. Longest Common Subsequence
	// string text1 = "abcde";
	// string text2 = "acd";
	// // string text2 = "";
	// cout << "String1: " << text1 << endl;
	// cout << "String2: " << text2 << endl;
	// cout << "Length of longest common subsequence is: " << solu.longestCommonSubsequence(text1, text2) << endl << endl;

	// 1092. Shortest Common Supersequence
	// string str1 = "cijkchc";
	// string str2 = "hcijkc";
	// cout << "String1: " << str1 << endl;
	// cout << "String2: " << str2 << endl;
	// cout << "Shortest Common Supersequence: " << solu.shortestCommonSupersequence(str1, str2) << endl << endl;

	/**
	 * TODO:
	 */
	// 1062. Longest Repeating Substring(lock)


	// 5. Longest Palindromic Substring
	// string s = "babad";
	// s = "atsgstsgstkpobbvijklmnnmlkjiqr";
	// cout << "String: " << s << endl;
	// cout << "Longest Palindromic Substring: " << solu.longestPalindrome(s) << endl << endl;

	// 516. Longest Palindromic Subsequence
	// string s = "babadbqwer";
	// s = "abcdefgecba";
	// // s = "bbbab";
	// cout << "String: " << s << endl;
	// cout << "Longest Palindromic Subsequence: " << solu.longestPalindromeSubseq(s) << endl << endl;

	// 583. Delete Operation for Two Strings
	// string word1 = "sea";
	// string word2 = "eat";
	// word1 = "intention";
	// word2 = "execution";
	// cout << "word1: " << word1 << endl;
	// cout << "word2: " << word2 << endl;
	// cout << "Minamal distance: " << solu.minDeleteDistance(word1, word2) << endl << endl;

	// 72. Edit Distance
	// string word1 = "horse";
	// string word2 = "ros";
	// word1 = "intention";
	// word2 = "execution";
	// cout << "word1: " << word1 << endl;
	// cout << "word2: " << word2 << endl;
	// cout << "Minimal edit distance: " << solu.minEditDistance(word1, word2) << endl << endl;

	/*
		0/1 Bag
		1. Iterates through the items should be in the outer cycle.
		2. Iterates through the bag should be in the inner cycle.
		3. The solutions could be always handled in an one-dimension array(dp[i]), and the bag iteration(inner cycle) should start from back to front in the array.
	*/

	// 416. Partition Equal Subset Sum
	// vector<int> nums = { 1, 5, 11, 5 };
	// nums = { 1,1,2,5,5,5,5 };
	// // nums = { 1, 3, 5 };
	// cout << "nums: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// cout << "Can partition: " << (solu.canPartition(nums) ? "true" : "false") << endl << endl;

	// 494. Target Sum
	// vector<int> nums = { 1, 5, 11, 5 };
	// // nums = { 1,1,2,5,5,5,5 };
	// // nums = { 1, 3, 5 };
	// nums = { 1, 1, 1, 1, 1 };
	// // nums = { 1, 0 };
	// cout << "nums: [ ";
	// printContainer(nums);
	// cout << " ]" << endl;
	// int S;
	// while (1)
	// {
	// 	cout << "Target Sum: ";
	// 	cin >> S;
	// 	cout << "Number of ways to get target sum: " << solu.findTargetSumWays(nums, S) << endl << endl;
	// }

	// AcWing 2. 01 Bag Question
	// solu.maxWorthFor01Bag();

	/*
		Complete Bag
	*/

	// 322. Coin Change
	// vector<int> coins = { 1, 2, 5 }; // 11->3
	// coins = { 186, 419, 83, 408 }; // 6249->20
	// // coins = {492,364,366,144,492,316,221,326,16,166,353}; // 5253
	// while (1)
	// {
	// 	int amount = 0;

	// 	while (amount <= 0)
	// 	{
	// 		cout << "amount: ";
	// 		cin >> amount;
	// 	}

	// 	cout << "Fewest number of coins: " << solu.coinChange(coins, amount) << endl << endl;
	// }

	// 518. Coin Change 2
	// vector<int> coins = { 1,2,5 }; // 5->4
	// // coins = { 2 }; // 3->0
	// coins = { 186,419,83,408 }; // 6249->19
	// while (1)
	// {
	// 	int amount = 0;

	// 	while (amount <= 0)
	// 	{
	// 		cout << "amount: ";
	// 		cin >> amount;
	// 	}

	// 	cout << "Number of combinations: " << solu.change(amount, coins) << endl << endl;
	// }

	// AcWing 3. Complete Bag Quesiton
	// solu.maxWorthForCompleteBag();

	/*
		Ordered Complete Bag
	*/

	// 139. Word Break
	// string s = "leetcode";
	// vector<string> wordDict = { "leet", "code" };
	// s = "applepenapple";
	// wordDict = { "apple", "pen" };
	// s = "catsandog";
	// wordDict = { "cats", "dog", "sand", "and", "cat" };
	// cout << "Can be segmented: " << (solu.wordBreak(s, wordDict) ? "true" : "false") << endl << endl;

	// 377. Combination Sum IV
	// vector<int> nums = { 1, 2, 3 };
	// while (1)
	// {
	// 	int target = 0;

	// 	while (target <= 0)
	// 	{
	// 		cout << "target: ";
	// 		cin >> target;
	// 	}

	// 	cout << "Number of combinations: " << solu.combinationSum4(nums, target) << endl << endl;
	// }

	/*
		Multiple Bag
	*/

	// AcWing 4. Multiple Bag Quesiton
	// solu.maxWorthForMultipleBag();

	// AcWing 5. Multiple Bag Question II
	// solu.maxWorthForMultipleBagII();

	// AcWing 6. Multiple Bag Question III
	// solu.maxWorthForMultipleBagIII();

	// AcWing 7. Compound Bag Question
	// solu.maxWorthForCompoundBag();

	// AcWing 8. Two Dimensional Bag Question --- V: volume limitation of bag, M: weight limitation of bag
	// solu.maxWorthForTwoDimensionalBag();

	// 474. Ones and Zeroes
	// vector<string> strs = { "10", "0001", "111001", "1", "0" };
	// while (1)
	// {
	// 	int m = -1;
	// 	int n = -1;

	// 	while (m < 0)
	// 	{
	// 		cout << "input number of 0, m: ";
	// 		cin >> m;
	// 	}

	// 	while (n < 0)
	// 	{
	// 		cout << "input number of 1, n: ";
	// 		cin >> n;
	// 	}

	// 	cout << "Maximum number of strings: " << solu.findMaxForm(strs, m, n) << endl << endl;
	// }

	// AcWing 9. Grouping Bag Question
	// solu.maxWorthForGroupingBag();

	// https://www.bilibili.com/video/av34467850/?p=2
	// AcWing 10. Dependence Bag Question --- Hard --- involve DFS

	// AcWing 11. Number of Solutions of Bag Question
	// solu.maxNumberOfSolutionsOfBag();

	// AcWing 12. Solution of Bag Question
	// solu.solutionOfBag();

	// 62. Unique Paths
	// while(1)
	// {
	// 	int m = 0;
	// 	int n = 0;

	// 	while (m <= 0)
	// 	{
	// 		cout << "m: ";
	// 		cin >> m;
	// 	}
	// 	while (n <= 0)
	// 	{
	// 		cout << "n: ";
	// 		cin >> n;
	// 	}

	// 	cout << "Number of unique paths: " << solu.uniquePaths(m, n) << endl << endl;
	// }

	// 64. Minimum Path Sum
	// vector<vector<int>> grid;
	// grid = {
	// 	{ 1,3,1 },
	// 	{ 1,5,1 },
	// 	{ 4,2,1 }
	// };
	// cout << "Grid:" << endl;
	// for(auto row : grid)
	// {
	// 	cout << "[ ";
	// 	printContainer(row);
	// 	cout << " ]" << endl;
	// }
	// cout << "Minimum Path Sum: " << solu.minPathSum(grid) << endl << endl;

	// 63. Unique Paths II
	// vector<vector<int>> obstacleGrid;
	// obstacleGrid = {
	// 	{ 0,0,0 },
	// 	{ 0,1,0 },
	// 	{ 0,0,0 }
	// };
	// cout << "obstacleGrid:" << endl;
	// for(auto row : obstacleGrid)
	// {
	// 	cout << "[ ";
	// 	printContainer(row);
	// 	cout << " ]" << endl;
	// }
	// cout << "Number of unique paths: " << solu.uniquePathsWithObstacles(obstacleGrid) << endl << endl;

	// 887. Super Egg Drop
	// while (1)
	// {
	// 	int K = 0, N = 0;
	// 	while (K <= 0)
	// 	{
	// 		cout << "Number of eggs: ";
	// 		cin >> K;
	// 	}
	// 	while (N <= 0)
	// 	{
	// 		cout << "Number of floors: ";
	// 		cin >> N;
	// 	}

	// 	cout << "Minimal number of moves: " << solu.superEggDrop(K, N) << endl << endl;
	// }

	// DOCK();

	return 0;
}

