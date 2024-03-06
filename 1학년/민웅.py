# 5557_1학년_1st-Grade
import sys
input = sys.stdin.readline

N = int(input())

n_lst = list(map(int, input().split()))

dp = [[0 for _ in range(21)] for _ in range(N-1)]
dp[0][n_lst[0]] += 1

for i in range(1, N-1):
    tmp = n_lst[i]
    for j in range(21):
        if dp[i-1][j]:
            if 0 <= j + tmp <= 20:
                dp[i][j + tmp] += dp[i-1][j]
            if 0 <= j - tmp <= 20:
                dp[i][j - tmp] += dp[i-1][j]

print(dp[N-2][n_lst[-1]])