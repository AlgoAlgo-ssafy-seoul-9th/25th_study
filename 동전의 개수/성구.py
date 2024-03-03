# 동전의 개수
# 108ms 24MB
import sys
input = sys.stdin.readline

# 입력
N, K = map(int, input().split())
coins = [int(input()) for _ in range(N)]

# settings
i = N-1     # 오름차순 입력이므로 역순 탐색
ans = 0

while K>0 and i>=0:
    coin = coins[i]
    # 코인 개수 체크
    ans += K // coin
    # 나머지 잔돈 체크
    K %= coin
    i-=1

print(ans)