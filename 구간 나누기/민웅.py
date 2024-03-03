# 2228_구간나누기_divide-sections
import sys
input = sys.stdin.readline

N, M = map(int, input().split())

n_lst = [int(input()) for _ in range(N)]


dp = [[[-float('inf') for _ in range(2)] for _ in range(M+1)] for _ in range(N+1)]

# 안 고른곳 초기화
for i in range(N+1):
    dp[i][0][0] = 0
    dp[i][0][1] = 0

# print(dp)

# N 번째의 숫자까지 고려할 때,
for i in range(1, N+1):
    tmp = n_lst[i-1]
    # j개 구간으로 나눈경우 각각 구간까지 최대 계산
    for j in range(1, M+1):
        # i번째 수 사용안함 -> 이전 수에서 j개의 구간을 고른경우 중, 이전 수를 사용한경우와 사용하지않은경우중 더 큰값
        dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1])
        # i번째 수 사용할거임 -> (현재 j번째 구간에 i번째수 포함시킨경우, i-1번째수에서 이전구간끝내고, 이번수를 새로운구간 시작으로 하는경우)
        dp[i][j][1] = max(dp[i-1][j][1] + tmp, dp[i-1][j-1][0] + tmp)
        if j > 1:
            # 2개 이상 구간일경우, j-1개의 구간을 선택한 경우에서 i-1 번째 수까지 중 최대값+현재 수로 j개의 구간을 선택한 배열을 갱신
            # i-1 번째 수까지인 이유 = 사이에 최소1개의 수를 건너뛰어야하기때문
            for k in range(1, i-1):
                dp[i][j][1] = max(dp[i][j][1], dp[k][j - 1][1] + tmp)


print(max(dp[-1][-1]))
