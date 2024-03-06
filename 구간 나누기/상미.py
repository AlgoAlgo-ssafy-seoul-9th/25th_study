import sys
input = sys.stdin.readline

N, M = map(int, input().split())
arr = []
for _ in range(N):
    arr.append(int(input()))
dp1 = [[-float('inf') for _ in range(M + 1)] for _ in range(N)]
dp2 = [[-float('inf') for _ in range(M + 1)] for _ in range(N)]
dp1[0][0] = 0       # dp1[i][j] : i번째 수를 포함하지 않는 j개 구간합 중 최댓값
dp2[0][1] = arr[0]  # dp2[i][j] : i번째 수를 포함하는 j개 구간합 중 최댓값

for i in range(1, N):
    dp1[i][0] = 0
    for j in range(1, M + 1):
        dp1[i][j] = max(dp1[i-1][j], dp2[i-1][j])
        dp2[i][j] = max(dp1[i-1][j-1] + arr[i], dp2[i-1][j] + arr[i])


print(max(dp1[N-1][M], dp2[N-1][M]))