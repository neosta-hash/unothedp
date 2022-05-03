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
		if (prices.size() <= 1)
			return 0;

		int buy = prices[0];
		int max_profit = 0;
		
		for (auto price : prices) {
			if (price < buy)
				buy = price; 
			else
				max_profit = max(max_profit, price - buy);
		}

		return max_profit;
	}

	// 122. Best Time to Buy and Sell Stock II
	int maxProfitII(vector<int>& prices) {
		size_t n = prices.size();
		if (n <= 1)
			return 0;
		
		int max_profit = 0;

		for (size_t i = 1; i < n; ++i)
			max_profit += max(0, prices[i] - prices[i - 1]);

		return max_profit;
	}

	// 309. Best Time to Buy and Sell Stock with Cooldown
	int maxProfitWithCooldown(vector<int>& prices) {
		size_t n = prices.size();
		if (n <= 1)
			return 0;

		vector<int> hold(n, 0);
		vector<int> sold(n, 0);

		hold[0] = -prices[0];
		hold[1] = max(hold[0], sold[0] - prices[1]);
		sold[1] = max(sold[0], hold[0] + prices[1]);

		for (size_t i = 2; i < n; ++i) {
			hold[i] = max(hold[i - 1], sold[i - 2] - prices[i]);
			sold[i] = max(sold[i - 1], hold[i - 1] + prices[i]);
		}

		return sold.back();
	}

	// 714. Best Time to Buy and Sell Stock with Transaction Fee
	int maxProfitWithTransactionFee(vector<int>& prices, int fee) {
		if (prices.size() <= 1)
			return 0;

		int hold = -prices[0] - fee;
		int sold = 0;

		for (auto price : prices) {
			int hold_tmp = hold;
			hold = max(hold, sold - price - fee);
			sold = max(sold, hold_tmp + price);
		}

		return sold;
	}

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	// int maxProfitAtMostTwoTransactions(vector<int>& prices) {
	// 	size_t n = prices.size();
	// 	if (n <= 1)
	// 		return 0;

	// 	vector<vector<int>> dp(n + 1, vector<int>(5, INT_MIN));

	// 	dp[0][0] = 0;

	// 	for (size_t i = 1; i <= n; ++i) {
	// 		for (size_t j = 0; j < 5; ++j) {
	// 			if (j % 2) {
	// 				// holding stock
	// 				dp[i][j] = dp[i - 1][j];
	// 				if (dp[i - 1][j - 1] != INT_MIN)
	// 					dp[i][j] = max(dp[i][j], dp[i - 1][ j - 1] - prices[i - 1]);
	// 			} else {
	// 				// not holding stock
	// 				if (j == 0)
	// 					dp[i][j] = 0;
	// 				else {
	// 					dp[i][j] = dp[i - 1][j];
	// 					if (dp[i - 1][j - 1] != INT_MIN)
	// 						dp[i][j] = max(dp[i][j], dp[i - 1][ j - 1] + prices[i - 1]);
	// 				}
	// 			}
	// 		}
	// 	}

	// 	int max_profit = dp[n][0];
	// 	for (size_t i = 2; i < 5; i += 2)
	// 		max_profit = max(max_profit, dp[n][i]);

	// 	return max_profit;
	// }

	// 123. Best Time to Buy and Sell Stock III (at Most Two Transactions)
	int maxProfitAtMostTwoTransactions(vector<int>& prices) {
		size_t n = prices.size();
		if (n <= 1)
			return 0;

		vector<int> dp1(5, INT_MIN);
		vector<int> dp2(5);

		dp1[0] = 0;

		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < 5; ++j) {
				dp2[j] = dp1[j];

				if (j % 2) {
					// holding stock
					if (dp1[j - 1] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j - 1] - prices[i]);
				} else {
					// not holding stock
					if (j > 0 && dp1[j - 1] != INT_MIN)
						dp2[j] = max(dp2[j], dp1[j - 1] + prices[i]);
				}
			}

			swap(dp1, dp2);
		}

		int max_profit = dp1[0];
		for (size_t i = 2; i < 5; i += 2)
			max_profit = max(max_profit, dp1[i]);

		return max_profit;
	}

	// 188. Best Time to Buy and Sell Stock IV (at Most K Transactions)
	int maxProfitAtMostKTransactions(int k, vector<int>& prices) {
		int n = prices.size();
		if (n <= 1 || k <= 0)
			return 0;
		
		size_t phases = 2 * k + 1;

		vector<int> dp1(phases, INT_MIN);
		vector<int> dp2(phases);

		dp1[0] = 0;

		for (size_t i = 0; i < n; ++i) {
			for (size_t j = 0; j < phases; ++j) {
				dp2[j] = dp1[j];

				if (j % 2 && dp1[j - 1] != INT_MIN) {
					dp2[j] = max(dp2[j], dp1[j - 1] - prices[i]);
				} else if (j > 0 && dp1[j - 1] != INT_MIN) {
					dp2[j] = max(dp2[j], dp1[j - 1] + prices[i]);
				}
			}

			swap(dp1, dp2);
		}

		int mp = dp1[0];
		for (size_t i = 2; i < phases; i += 2)
			mp = max(mp, dp1[i]);

		return mp;
	}

	// 70. Climbing Stairs
	int climbStairs(int n) {
		vector<int> dp = {0, 1, 2};

		for (size_t i = dp.size(); i <= n; ++i)
			dp.emplace_back(dp[i - 1] + dp[i - 2]);

		return dp[n];
	}

	// 746. Min Cost Climbing Stairs
	int minCostClimbingStairs(vector<int>& cost) {
		size_t n = cost.size();

		if (n == 0)
			return 0;
		else if (n == 1)
			return cost[0];
		else if (n == 2)
			return min(cost[0], cost[1]);

		vector<int> dp1 = {cost[0], cost[0] + cost[1]}; // from step 0
		vector<int> dp2 = {cost[0] + cost[1], cost[1]}; // from step 1

		for (int i = 2; i < n; ++i) {
			dp1.emplace_back(cost[i] + min(dp1[i-1], dp1[i-2]));
			dp2.emplace_back(cost[i] + min(dp2[i-1], dp2[i-2]));
		}

		return min(min(dp1[n-1], dp1[n-2]), min(dp2[n-1], dp2[n-2]));
	}

	// 413. Arithmetic Slices
	int numberOfArithmeticSlices(vector<int>& nums) {
		size_t n = nums.size();

		if (n <= 2)
			return 0;

		vector<size_t> dp(n, 0);
		int count = 0;

		for (size_t i = 2; i < n; ++i) {
			if (nums[i] - nums[i-1] == nums[i-1] - nums[i-2]) {
				dp[i] = dp[i-1] + 1;
				count += dp[i];
			}
		}

		return count;
	}

	// 300. Longest Increasing Subsequence
	int lengthOfLIS(vector<int>& nums) {
		size_t n = nums.size();

		if (n < 2)
			return n;

		vector<int> dp(n, 1);
		int len = 1;

		for (int i = 1; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				if (nums[i] > nums[j]) {
					dp[i] = max(dp[i], dp[j] + 1);
				}
			}
			len = max(len, dp[i]);
		}

		return len;
	}

	/**
	 * TODO: find lengthOfLIS by using binary search
	 */

	// 646. Maximum Length of Pair Chain
	static bool cmp_pairs(vector<int>& p1, vector<int>p2) {
		return p1[0] < p2[0];
	}

	int findLongestChain(vector<vector<int>>& pairs) {
		int n = pairs.size();

		if (n < 2)
			return n;
		
		vector<int> dp(n ,1);
		int len = 1;

		sort(pairs.begin(), pairs.end(), cmp_pairs);

		for (int i = 1; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				if (pairs[i][0] > pairs[j][1])
					dp[i] = max(dp[i], dp[j] + 1);
			}
			len = max(len, dp[i]);
		}

		return len;
	}

	// 376. Wiggle Subsequence
	// O(n^2)
	// int wiggleMaxLength(vector<int>& nums) {
	// 	size_t n = nums.size();

	// 	if (n == 0)
	// 		return 0;

	// 	vector<int> lw(n, 1);
	// 	vector<int> rw(n, 1);
	// 	int len = 1;

	// 	for (int i = 1; i < n; ++i) {
	// 		for (int j = i - 1; j >= 0; --j) {
	// 			if (nums[i] > nums[j])
	// 				rw[i] = max(rw[i], lw[j] + 1);
	// 			else if (nums[i] < nums[j])
	// 				lw[i] = max(lw[i], rw[j] + 1);
	// 		}
	// 		len = max(len, max(lw[i], rw[i]));
	// 	}

	// 	return len;
	// }

	// O(n)
	int wiggleMaxLength(vector<int>& nums) {
		size_t n = nums.size();

		if (n == 0)
			return 0;

		int lw = 1;
		int rw = 1;

		for (int i = 1; i < n; ++i) {
			if (nums[i] > nums[i - 1])
				rw = lw + 1;
			else if (nums[i] < nums[i - 1])
				lw = rw + 1;
		}

		return max(lw, rw);
	}

	// 198. House Robber
	int rob(vector<int>& nums) {
		int n = nums.size();

		if (n  == 0)
			return 0;
		else if (n == 1)
			return nums.front();
		else if (n == 2)
			return max(nums.front(), nums.back());

		vector<int> dp(n, 0);

		dp[0] = nums[0];
		dp[1] = nums[1];
		dp[2] = nums[0] + nums[2];

		for (int i = 3; i < n; ++i)
			dp[i] = nums[i] + max(dp[i - 2], dp[i - 3]);

		return max(dp[n - 1], dp[n - 2]);
	}

	// 213. House Robber II
	int robII(vector<int>& nums) {
		int n = nums.size();

		if (n == 0)
			return 0;
		else if (n == 1)
			return nums.front();
		else if (n == 2)
			return max(nums.front(), nums.back());

		vector<int> dp1(n, 0);
		vector<int> dp2(n, 0);

		dp1[0] = nums[0];
		dp1[1] = INT_MIN; 
		dp1[2] = nums[0] + nums[2];

		dp2[0] = INT_MIN;
		dp2[1] = nums[1];
		dp2[2] = nums[2];

		for (int i = 3; i < n; ++i) {
			dp1[i] = nums[i] + max(dp1[i-2], dp1[i-3]);
			dp2[i] = nums[i] + max(dp2[i-2], dp2[i-3]);
		}

		return max(max(dp1[n-2], dp1[n-3]), max(dp2[n-1], dp2[n-2]));
	}

	// 53. Maximum Subarray
	int maxSubArray(vector<int>& nums) {
		int n = nums.size();

		if (n == 0)
			return 0;

		int sum = nums[0];
		int max_sum = nums[0];

		for (int i = 1; i < n; ++i) {
			sum = max(nums[i], sum + nums[i]);
			max_sum = max(max_sum, sum);
		}

		return max_sum;
	}

	// 650. 2 Keys Keyboard
	int minStepsWith2KeysKeyboard(int n) {
		vector<int> dp(n + 1, 0);

		for (int i = 2; i <= n; ++i) {
			dp[i] = i;

			for (int j = 2; j <= i/2; ++j) {
				if (i % j == 0)
					dp[i] = min(dp[i], dp[j] + i/j);
			}
		}

		return dp[n];
	}

	// 1218. Longest Arithmetic Subsequence of Given Difference
	/**
	 * TODO: submitted an answer(N^2) that exceeds the time limitation, need to solve it using binary search
	 */
	int longestSubsequence(vector<int>& arr, int difference) {
		unordered_map<int, int> dp;
		int len = 1;

		for (auto num : arr) {
			int expect = num + difference;

			if (dp[num]) {
				dp[expect] = dp[num] + 1;
				len = max(len, dp[expect]);

			} else {
				dp[expect] = 1;
			}
		}

		return len;
	}

	// 392. Is Subsequence
	bool isSubsequence(string s, string t) {
		int pos = 0;

		for (char c : s) {
			pos = t.find(c, pos);
			if (pos == string::npos)
				return false;
			++pos;
		}

		return true;
	}

	// 1143. Longest Common Subsequence
	// int longestCommonSubsequence(string text1, string text2) {
	// 	int n1 = text1.size();
	// 	int n2 = text2.size();

	// 	vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

	// 	for (int i = 1; i <= n1; ++i) {
	// 		for (int j = 1; j <= n2; ++j) {
	// 			if (text1[i - 1] == text2[j - 1])
	// 				dp[i][j] = dp[i - 1][j - 1] + 1;
	// 			else
	// 				dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
	// 		}
	// 	}

	// 	return dp[n1][n2];
	// }

	int longestCommonSubsequence(string text1, string text2) {
		int n1 = text1.size();
		int n2 = text2.size();

		vector<int> dp1(n2 + 1, 0);
		vector<int> dp2(n2 + 1, 0);

		for (int i = 1; i <= n1; ++i) {
			for (int j = 1; j <= n2; ++j) {
				if (text1[i - 1] == text2[j - 1])
					dp2[j] = dp1[j - 1] + 1;
				else
					dp2[j] = max(dp1[j], dp2[j - 1]);
			}

			swap(dp1, dp2);
		}

		return dp1[n2];
	}

	// 1092. Shortest Common Supersequence
	string shortestCommonSupersequence(string str1, string str2) {
		int n1 = str1.size();
		int n2 = str2.size();

		if (n1 == 0)
			return str2;
		else if (n2 == 0)
			return str1;

		vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

		for (int i = 1; i <= n1; ++i) {
			for (int j = 1; j <= n2; ++j) {
				if (str1[i - 1] == str2[j - 1])
					dp[i][j] = dp[i - 1][j - 1] + 1;
				else
					dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
			}
		}

		if (dp[n1][n2] == 0)
			return str1 + str2;

		int i = n1;
		int j = n2;
		deque<char> scs;
		char c;

		while (i || j) {
			if (!i)
				c = str2[--j];
			else if (!j)
				c = str1[--i];
			else if (str1[i - 1] == str2[j - 1])
				c = str1[--i] = str2[--j];
			else if (dp[i][j] == dp[i - 1][j])
				c = str1[--i];
			else if (dp[i][j] == dp[i][j - 1])
				c = str2[--j];
			
			scs.push_front(c);
		}

		return {scs.begin(), scs.end()};
	}

	// 5. Longest Palindromic Substring
	int isPalindrome(int l, int r, string &s) {
		while (l <= r && l >= 0 && r < s.size() && s[l] == s[r])
			--l, ++r;

		return r - l - 1;
	}

	string longestPalindrome(string s) {
		int n = s.size();
		if (n == 0)
			return s;

		int start_pos = 0;
		int max_len = 0;

		for (int i = 0; i < n; ++i) {
			if (max_len >= 2*(n - i) - 1)
				break;

			int len = max(isPalindrome(i, i, s), isPalindrome(i, i + 1, s));
			if (len > max_len) {
				max_len = len;
				start_pos = i - (len - 1)/2;
			}
		}

		return s.substr(start_pos, max_len);
	}

	// 516. Longest Palindromic Subsequence
	/**
	 * The solutions what I have done in the master and lap2, I cannot understand them now,
	 * maybe I didn't have a perfect understanding for the logic at that time as well, so
	 * I finally made a decision that I will solve the quesiton in the future interview by
	 * remembering the code.
	 * But at this time, I learned another solution that is more commonly used by other
	 * programmers(Huahuajiang, .etc), it carries a different logic that would result in
	 * a new code implementation which is differing with my previous code, but at least
	 * I can understand what the code is talking about.
	 */
	// int longestPalindromeSubseq(string s) {
	// 	int n = s.size();

	// 	vector<vector<int>> dp(n, vector<int>(n, 0));

	// 	for (int len = 1; len <= n; ++len) {
	// 		for (int i = 0; i + len <= n; ++i) {
	// 			int j = i + len - 1;

	// 			if (i == j) {
	// 				dp[i][j] = 1;
	// 				continue;
	// 			}

	// 			if (s[i] == s[j])
	// 				dp[i][j] = dp[i + 1][j - 1] + 2; // len = l - 2
	// 			else
	// 				dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]); // len = l - 1
	// 		}
	// 	}

	// 	return dp[0][n - 1];
	// }

	int longestPalindromeSubseq(string s) {
		int n = s.size();

		vector<int> dp0(n, 0);	// solutions for len = l
		vector<int> dp1(n, 0);	// solutions for len = l - 1
		vector<int> dp2(n, 0);  // solutions for len = l - 2

		for (int len = 1; len <= n; ++len) {
			for (int i = 0; i + len <= n; ++i){
				int j = i + len - 1;
				if (i == j) {
					dp0[i] = 1;
					continue;
				}

				if (s[i] == s[j])
					dp0[i] = dp2[i + 1] + 2;
				else
					dp0[i] = max(dp1[i], dp1[i + 1]);
			}

			swap(dp0, dp1);
			swap(dp0, dp2);
		}

		return dp1[0];
	}

	// 583. Delete Operation for Two Strings
	// minimum number of deletions = (n1 - lcs) + (n2 - lcs) = n1 + n2 - 2lcs
	int minDeleteDistance(string word1, string word2) {
		int n1 = word1.size();
		int n2 = word2.size();

		if (n1 == 0)
			return n2;
		else if (n2 == 0)
			return n1;

		vector<int> dp1(n2 + 1, 0);
		vector<int> dp2(n2 + 1, 0);

		for (int i = 1; i <= n1; ++i) {
			for (int j = 1; j <= n2; ++j) {
				if (word1[i - 1] == word2[j - 1])
					dp2[j] = dp1[j - 1] + 1;
				else
					dp2[j] = max(dp1[j], dp2[j - 1]);
			}
			swap(dp1, dp2);
		}

		return n1 + n2 - 2*dp1[n2];
	}

	// 72. Edit Distance
	int minEditDistance(string word1, string word2) {
		int n1 = word1.size();
		int n2 = word2.size();

		if (n1 == 0)
			return n2;
		else if (n2 == 0)
			return n1;

		vector<vector<int>> dp(n1, vector<int>(n2, 0));

		for (int i = 0; i < n1; ++i) {
			if (word1[i] == word2[0]) {
				dp[i][0] = i;
			} else {
				if (i == 0)
					dp[i][0] = 1;
				else
					dp[i][0] = dp[i - 1][0] + 1;
			}
		}

		for (int j = 0; j < n2; ++j) {
			if (word2[j] == word1[0]) {
				dp[0][j] = j;
			} else {
				if (j == 0)
					dp[0][j] = 1;
				else
					dp[0][j] = dp[0][j - 1] + 1;
			}
		}

		for (int i = 1; i < n1; ++i) {
			for (int j = 1; j < n2; ++j) {
				if (word1[i] == word2[j])
					dp[i][j] = dp[i - 1][j - 1];
				else
					dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
			}
		}

		return dp[n1 - 1][n2 - 1];
	}

	// 416. Partition Equal Subset Sum
	// bool canPartition(vector<int>& nums) {
	// 	int n = nums.size();
	// 	int sum = 0;

	// 	for (auto num : nums)
	// 		sum += num;

	// 	if (sum % 2)
	// 		return false;

	// 	sum /= 2;

	// 	vector<vector<bool>> dp(n + 1, vector<bool>(sum + 1, false));

	// 	dp[0][0] = true;

	// 	for (int i = 1; i <= n; ++i) {
	// 		dp[i] = dp[i - 1];
	// 		for (int j = nums[i - 1]; j <= sum; ++j) {
	// 			if (dp[i - 1][j - nums[i - 1]]) {
	// 				dp[i][j] = true;
	// 				if (j == sum)
	// 					return true;
	// 			}
	// 		}
	// 	}

	// 	return false;
	// }

	bool canPartition(vector<int>& nums) {
		int n = nums.size();
		int sum = 0;

		for (auto num : nums)
			sum += num;
		
		if (sum % 2)
			return false;

		sum /= 2;

		vector<bool> dp(sum + 1, false);
		dp[0] = true;

		for (auto num : nums) {
			if (num > sum)
				return false;

			for (int i = sum; i >= num; --i) {
				if (dp[i - num]) {
					dp[i] = true;
					if (i == sum)
						return true;
				}
			}
		}

		return false;
	}

	// 494. Target Sum
	int findTargetSumWays(vector<int>& nums, int target) {
		if (nums.size() == 0)
			return 0;

		int sum = accumulate(nums.begin(), nums.end(), 0);
		if (abs(target) > sum)
			return 0;

		int bags = sum * 2 + 1;

		vector<int> dp1(bags, 0);
		vector<int> dp2(bags, 0);
		dp1[sum] = 1;

		for (int num : nums) {
			for (int i = 0; i < bags; ++i) {
				dp2[i] = 0;
				if (i - num >= 0)
					dp2[i] += dp1[i - num];
				if (i + num < bags)
					dp2[i] += dp1[i + num];
			}
			swap(dp1, dp2);
		}

		return dp1[target + sum];
	}

	// AcWing 2. 01 Bag Question
	int maxWorthFor01Bag() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0); 

		for (int i = 0; i < N; ++i) {
			int v, w;
			cin >> v >> w;
		
			for (int j = V; j >= v; --j) {
				dp[j] = max(dp[j], dp[j - v] + w);
			}
		}

		cout << dp[V];

		return 0;
	}

	// 322. Coin Change
	/**
	 * Time Limit Exceeded
	 */
	// int coinChange(vector<int>& coins, int amount) {
	// 	int n = coins.size();

	// 	vector<vector<int>> dp(n + 1, vector<int>(amount + 1, INT_MAX));

	// 	dp[0][0] = 0;

	// 	for (int i = 1; i <= n; ++i) {
	// 		int coin = coins[i - 1];
	// 		for (int j = 0; j <= amount; ++j) {
	// 			dp[i][j] = dp[i - 1][j];
	// 			if (j < coin)
	// 				continue;

	// 			for (int k = 1; k <= j / coin; ++k) {
	// 				int x = j - coin * k;

	// 				if (dp[i - 1][x] != INT_MAX)
	// 					dp[i][j] = min(dp[i][j], dp[i - 1][x] + k);
	// 			}
	// 		}
	// 	}

	// 	return dp[n][amount] == INT_MAX ? -1 : dp[n][amount];
	// }

	int coinChange(vector<int>& coins, int amount) {
		vector<int> dp(amount + 1, INT_MAX);
		dp[0] = 0;

		for (int coin : coins) {
			for (int i = coin; i <= amount; ++i) {
				if (dp[i - coin] != INT_MAX)
					dp[i] = min(dp[i], dp[i - coin] + 1);
			}
		}

		return dp[amount] == INT_MAX ? -1 : dp[amount];
	}

	// 518. Coin Change 2
	// Find the number of solutions of complete bag question
	int change(int amount, vector<int>& coins) {
		vector<int> dp(amount + 1, 0);
		dp[0] = 1;

		for (int coin : coins) {
			for (int i = coin; i <= amount; ++i) {
				if (dp[i - coin])
					dp[i] += dp[i - coin];
			}
		}

		return dp[amount];
	}

	// AcWing 3. Complete Bag Quesiton
	int maxWorthForCompleteBag() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0);

		for (int i = 0; i < N; ++i) {
			int v, w;
			cin >> v >> w;

			for (int j = v; j <= V; ++j){
				dp[j] = max(dp[j], dp[j - v] + w);
			}
		}

		cout << dp[V];

		return 0;
	}

	// 139. Word Break
	bool wordBreak(string s, vector<string>& wordDict) {
		int n = s.length();

		vector<bool> dp(n + 1, false);
		dp[0] = true;

		for (int i = 1; i <= n; ++i) {
			for (auto word : wordDict) {
				int len = word.length();

				// my code
				if (i >= len && dp[i - len] &&
				    s.substr(i - len, len) == word)
					dp[i] = true;

				// other people's code
				// if (i >= len && s.substr(i - len, len) == word)
				// 	dp[i] = dp[i] || dp[i - len];
			}
		}

		return dp[n];
	}

	// 377. Combination Sum IV
	// Find the number of solutions of ordered complete bag question
	int combinationSum4(vector<int>& nums, int target) {
		vector<unsigned int> dp(target + 1, 0);
		dp[0] = 1;

		for (int i = 1; i <= target; ++i) {
			for (auto num : nums) {
				if (i >= num && dp[i - num])
					dp[i] += dp[i - num];
			}
		}

		return dp[target];
	}

	// AcWing 4. Multiple Bag Quesiton
	int maxWorthForMultipleBag() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0);

		for (int i = 0; i < N; ++i) {
			int v, w, s;
			cin >> v >> w >> s;

			for (int j = V; j >= v; --j) {
				for (int k = 1; k * v <= j && k <= s; ++k)
					dp[j] = max(dp[j], dp[j - k * v] + k * w);
			}
		}

		cout << dp[V] << endl;

		return 0;
	}

	// AcWing 5. Multiple Bag Question II
	// struct item {
	// 	int v;
	// 	int w;
	// };

	// int maxWorthForMultipleBagII() {
	// 	int N, V;
	// 	cin >> N >> V;

	// 	vector<struct item> items;
	// 	vector<int> dp(V + 1, 0);

	// 	for (int i = 0; i < N; ++i) {
	// 		int v, w, s;
	// 		cin >> v >> w >> s;

	// 		for (int j = 1; j <= s; j *= 2) {
	// 			s -= j;
	// 			items.push_back({v * j, w * j});
	// 		}
	// 		if (s)
	// 			items.push_back({v * s, w * s});
	// 	}

	// 	for (int i = 0; i < items.size(); ++i) {
	// 		for (int j = V; j >= items[i].v; --j)
	// 			dp[j] = max(dp[j], dp[j - items[i].v] + items[i].w);
	// 	}

	// 	cout << dp[V] << endl;

	// 	return 0;
	// }

	int maxWorthForMultipleBagII() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0);

		for (int i = 0; i < N; ++i) {
			int v, w, s;
			cin >> v >> w >> s;

			for (int k = 1; k <= s; k *= 2) {
				s -= k;
				for (int j = V; j >= v * k; --j)
					dp[j] = max(dp[j], dp[j - v * k] + w * k);
			}
			if (s) {
				for (int j = V; j >= v * s; --j)
					dp[j] = max(dp[j], dp[j - v * s] + w * s);
			}
		}

		cout << dp[V] << endl;

		return 0;
	}

	// AcWing 7. Compound Bag Question
	int maxWorthForCompoundBag() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0);

		for (int i = 0; i < N; ++i) {
			int v, w, s;
			cin >> v >> w >> s;

			if (s == -1) {
				for (int j = V; j >= v; --j)
					dp[j] = max(dp[j], dp[j - v] + w);
			} else if (s == 0) {
				for (int j = v; j <= V; ++j)
					dp[j] = max(dp[j], dp[j - v] + w);
			} else if (s > 0) {
				for (int k = 1; k <= s; k *= 2) {
					s -= k;
					for (int j = V; j >= k * v; --j)
						dp[j] = max(dp[j], dp[j - k * v] + k * w);
				}
				if (s) {
					for (int j = V; j >= s * v; --j)
						dp[j] = max(dp[j], dp[j - s * v] + s * w);
				}
				
			}
		}

		cout << dp[V] << endl;

		return 0;
	}

	// AcWing 8. Two Dimensional Bag Question --- V: volume limitation of bag, M: weight limitation of bag
	int maxWorthForTwoDimensionalBag() {
		int N, V, M;
		cin >> N >> V >> M;

		vector<vector<int>> dp(V + 1, vector<int>(M + 1, 0));

		for (int i = 0; i < N; ++i) {
			int v, m, w;
			cin >> v >> m >> w;

			for (int j = V; j >= v; --j) {
				for (int k = M; k >= m; --k)
					dp[j][k] = max(dp[j][k], dp[j - v][k - m] + w); 
			}
		}

		cout << dp[V][M];

		return 0;
	}

	// 474. Ones and Zeroes
	// Two Dimensional Bag Question
	int findMaxForm(vector<string>& strs, int m, int n) {
		if (strs.size() == 0)
			return 0;

		vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

		for (auto str : strs) {
			int num0 = count(str.begin(), str.end(), '0');
			int num1 = count(str.begin(), str.end(), '1');

			for (int i = m; i >= num0; --i) {
				for (int j = n; j >= num1; --j) {
					dp[i][j] = max(dp[i][j], dp[i - num0][j - num1] + 1);
				}
			}
		}

		return dp[m][n];
	}

	// AcWing 9. Grouping Bag Question
	int maxWorthForGroupingBag() {
		int N, V;
		cin >> N >> V;

		vector<int> dp(V + 1, 0);

		for (int i = 0; i < N; ++i) {
			int s;
			cin >> s;

			vector<int> vs(s, 0);
			vector<int> ws(s, 0);

			for (int j = 0; j < s; ++j)
				cin >> vs[j] >> ws[j];

			for (int j = V; j >= 0; --j) {
				for (int k = 0; k < s; ++k) {
					if (j >= vs[k])
						dp[j] = max(dp[j], dp[j - vs[k]] + ws[k]);
				}
			}
		}

		cout << dp[V] << endl;

		return 0;
	}

	// AcWing 11. Number of Solutions of Bag Question
	int maxNumberOfSolutionsOfBag() {
		int mod = 1e9 + 7;

		int N, V;
		cin >> N >> V;

		vector<int> W(V + 1, 0);
		vector<int> S(V + 1, 0);
		S[0] = 1;

		for (int i = 0; i < N; ++i) {
			int v, w;
			cin >> v >> w;

			for (int j = V; j >= v; --j) {
				int worth = max(W[j], W[j - v] + w);
				// if (worth == W[j - v] + w) {
				// 	if (worth == W[j])
				// 		S[j] = (S[j] + S[j - v]) % mod;
				// 	else
				// 		S[j] = S[j - v];
				// }
				// W[j] = worth;
				int slnum = 0;
				if (worth == W[j])
					slnum = S[j];
				if (worth == W[j - v] + w)
					slnum = (slnum + S[j - v]) % mod;

				W[j] = worth;
				S[j] = slnum;
			}
		}

		int maxw = *max_element(W.begin(), W.end());
		int sn = 0;

		for (int i = 0; i <= V; ++i) {
			if (W[i] == maxw)
				sn += S[i];
		}

		cout << sn << endl;

		return 0;
	}

	// AcWing 12. Solution of Bag Question
	int solutionOfBag() {
		int N, V;
		cin >> N >> V;

		vector<int> v(N, 0);
		vector<int> w(N, 0);

		for (int i = 0; i < N; ++i)
			cin >> v[i] >> w[i];

		vector<vector<int>> dp(N + 1, vector<int>(V + 1, 0));

		for (int i = N - 1; i >= 0; --i) {
			for (int j = 1; j <= V; ++j) {
				dp[i][j] = dp[i + 1][j];
				if (j >= v[i])
					dp[i][j] = max(dp[i][j], dp[i + 1][j - v[i]] + w[i]);
			}
		}

		for (int i = 0; i < N; ++i) {
			if (V >= v[i] && dp[i][V] == dp[i + 1][V - v[i]] + w[i]) {
				cout << i + 1 << " ";
				V -= v[i];
			}
		}

		return 0;
	}

	// 62. Unique Paths
	int uniquePaths(int m, int n) {
		vector<int> dp1(n + 1, 0);
		vector<int> dp2(n + 1, 0);

		dp1[1] = 1;

		for (int i = 0; i < m; ++i) {
			for (int j = 1; j <= n; ++j) {
				dp2[j] = dp2[j - 1] + dp1[j];
			}
			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 64. Minimum Path Sum
	int minPathSum(vector<vector<int>>& grid) {
		int m = grid.size();
		int n = grid.front().size();

		vector<int> dp1(n + 1, INT_MAX);
		vector<int> dp2(n + 1, INT_MAX);

		dp1[1] = 0;

		for (int i = 0; i < m; ++i) {
			for (int j = 1; j <= n; ++j) {
				dp2[j] = min(dp2[j - 1], dp1[j]) + grid[i][j - 1];
			}

			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 63. Unique Paths II
	int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
		int m = obstacleGrid.size();
		int n = obstacleGrid.front().size();

		vector<int> dp1(n + 1, 0);
		vector<int> dp2(n + 1, 0);

		dp1[1] = 1;

		for (int i = 0; i < m; ++i) {
			for (int j = 1; j <= n; ++j) {
				if (obstacleGrid[i][j - 1])
					dp2[j] = 0;
				else
					dp2[j] = dp2[j - 1] + dp1[j];
			}
			swap(dp1, dp2);
		}

		return dp1[n];
	}

	// 887. Super Egg Drop
	// int superEggDrop(int K, int N) {
	// 	if (1 == K || N < 3)
	// 		return N;

	// 	vector<int> dp1(N+1, 0);
	// 	vector<int> dp2(N+1, 0);

	// 	for (int i = 1; i <= N; i++)
	// 		dp1[i] = i;

	// 	for (int k = 2; k <= K; k++)
	// 	{
	// 		for (int m = k; m <= N; m++)
	// 		{
	// 			if(m == k)
	// 				dp2[m] = dp1[m-1]*2 + 1;
	// 			else
	// 				dp2[m] = dp1[m-1] + dp2[m-1] + 1;
				
	// 			if (dp2[m] >= N)
	// 			{
	// 				if(k == K || m == k)
	// 					return m;
	// 				break;
	// 			}
	// 		}

	// 		swap(dp1, dp2);
	// 	}

	// 	return 0;
	// }

	int superEggDrop(int k, int n) {
		if (k == 1 || n == 1)
			return n;

		vector<int> dp1(n + 1, 0);
		vector<int> dp2(n + 1, 0);

		for (int i = 0; i <= n; ++i)
			dp1[i] = i;

		for (int i = 2; i <= k; ++i) {
			for (int j = i; j <= n; ++j) {
				if (j == i)
					dp2[j] = dp1[j - 1] * 2 + 1;
				else
					dp2[j] = dp1[j - 1] + dp2[j - 1] + 1;

				if (dp2[j] >= n) {
					if (i == k || i == j)
						return j;
					break;
				}
			}

			swap(dp1, dp2);
		}

		return 0;
	}
};

// 303. Range Sum Query - Immutable
class NumArray {
public:
	vector<int> dp;

	NumArray(vector<int>& nums) {
		dp.emplace_back(0);

		for (auto num : nums)
			dp.emplace_back(dp.back() + num);
	}

	int sumRange(int left, int right) {
		return (dp[right + 1] - dp[left]);
	}
};

// 304. Range Sum Query 2D - Immutable
class NumMatrix {
public:
	vector<vector<int>> dp;

	NumMatrix(vector<vector<int>>& matrix) {
		int row = matrix.size();
		int col = matrix[0].size();

		dp.resize(row + 1, vector<int>(col + 1, 0));

		for (int i = 1; i <= row; ++i)
			for (int j = 1; j <= col; ++j)
				dp[i][j] = dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1] + matrix[i-1][j-1];
	}

	int sumRegion(int row1, int col1, int row2, int col2) {
		return dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1] + dp[row1][col1];
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
	// vector<vector<int>> pairs;
	// pairs = {
	// 	{ 1,2 },
	// 	{ 2,3 },
	// 	{ 3,4 }
	// };

	// pairs = {
	// 	{ 3,4 },
	// 	{ 2,3 },
	// 	{ 1,2 }
	// };

	// cout << "Length of the longest chain is: " << solu.findLongestChain(pairs) << endl << endl;

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
	solu.solutionOfBag();

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

