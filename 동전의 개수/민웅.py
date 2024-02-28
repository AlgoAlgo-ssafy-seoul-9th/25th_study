import sys
input = sys.stdin.readline

N, K = map(int, input().split())

coins = []

for _ in range(N):
    coins.append(int(input()))

ans = 0
cnt = 0
for i in range(N-1, -1, -1):
    tmp = coins[i]
    while True:
        if ans + tmp > K:
            break
        ans += tmp
        cnt += 1

print(cnt)