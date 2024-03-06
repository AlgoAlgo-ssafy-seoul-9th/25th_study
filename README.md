# 25th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [구간 나누기](https://www.acmicpc.net/problem/2228)

### [민웅](./구간%20나누기/민웅.py)

```py
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

```

### [상미](./구간%20나누기/상미.py)

```py

```

### [성구](./구간%20나누기/성구.py)

```py

```

### [승우](./구간%20나누기/승우.py)

```py


```

## [1학년](https://www.acmicpc.net/problem/5557)

### [민웅](./1학년/민웅.py)

```py
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
```

### [상미](./1학년/상미.py)

```py
import sys
input = sys.stdin.readline

n = int(input())
arr = list(map(int, input().split()))

dp = [[0] * 21 for _ in range(n)]

# 첫 번째 수는 무조건 저장
dp[0][arr[0]] = 1

for i in range(1, n - 1):
    for j in range(21):
        if dp[i - 1][j]:
            if j + arr[i] <= 20:
                # 더하기
                dp[i][j + arr[i]] += dp[i - 1][j]
            if j - arr[i] >= 0:
                # 빼기
                dp[i][j - arr[i]] += dp[i - 1][j]

print(dp[n-2][arr[-1]])
```

### [성구](./1학년/성구.py)

```py
# 5557 1학년
# 31120 KB 40 ms
import sys
input = sys.stdin.readline


def solution():
    # 입력
    # arr를 변환시키지 않으니 tuple로 받는게 메모리 이득!
    N = int(input())
    arr = tuple(map(int, input().split()))

    dp = [[0] * 21 for _ in range(N-1)]
    # 초깃값 체크(첫번째 값에 1개 체크)
    dp[0][arr[0]] = 1
    # 문제를 제대로 읽자!
    # 중간 숫자는 0~20 사이 숫자(양 끝 값 포함)
    for i in range(1, N-1):
        # 모든 값 확인
        for j in range(21):
            if dp[i-1][j]:
                # 덧셈의 경우
                if j + arr[i] <= 20:
                    dp[i][j+arr[i]] += dp[i-1][j]
                # 뺄셈의 경우
                if j - arr[i] >= 0:
                    dp[i][j-arr[i]] += dp[i-1][j]
    # 마지막은 "="이므로 마지막 값(인덱스) 확인
    print(dp[-1][arr[-1]])

if __name__ == "__main__":
    solution()
```

### [승우](./1학년/승우.py)

```py
import sys
input = sys.stdin.readline

N = int(input())
numbers = list(map(int, input().split()))

dp = [[0 for _ in range(21)] for _ in range(N)]
dp[0][numbers[0]] = 1

for i in range(1, N - 1):
    for j in range(21):
        if dp[i-1][j]:
            if j + numbers[i] <= 20:
                dp[i][j+numbers[i]] += dp[i - 1][j]
            if j - numbers[i] >=0:
                dp[i][j-numbers[i]] += dp[i - 1][j]

print(dp[N-2][numbers[N-1]])

```

<br/>

</details>

<br/><br/>

# 지난주 스터디 문제

<details markdown="1">
<summary>접기/펼치기</summary>

## [동전의 개수](https://www.codetree.ai/problems/number-of-coins/description)

### [민웅](./동전의%20개수/민웅.py)

```py
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
```

### [상미](./동전의%20개수/상미.py)

```py

```

### [성구](./동전의%20개수/성구.py)

```py
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
```

### [승우](./동전의%20개수/승우.py)

```py


```

## [곰돌이의 모험](https://www.codetree.ai/problems/adventure-of-teddy-bear/description)

### [민웅](./곰돌이의%20모험/민웅.py)

```py
import sys
from itertools import product

input = sys.stdin.readline
dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def bt(loc, score, V, time):
    global ans
    if time == 3:
        if score > ans:
            ans = score
        return

    tmp_loc = [[] for _ in range(len(loc))]
    for i in range(len(loc)):
        x, y = loc[i]

        for d in dxy:
            nx = x + d[0]
            ny = y + d[1]

            if 0 <= nx <= N - 1 and 0 <= ny <= N - 1:
                if field[nx][ny] != -1:
                    tmp_loc[i].append([nx, ny])
    idx_check = []

    for i in range(len(loc)):
        if tmp_loc[i]:
            idx_check.append(i)
    idx_cnt = len(idx_check)
    if idx_cnt == 3:
        prod = product(tmp_loc[idx_check[0]], tmp_loc[idx_check[1]], tmp_loc[idx_check[2]])
    elif idx_cnt == 2:
        prod = product(tmp_loc[idx_check[0]], tmp_loc[idx_check[1]])
    elif idx_cnt == 1:
        prod = product(tmp_loc[idx_check[0]])
    else:
        return

    for value in prod:
        new_V = [[V[i][j] for j in range(N)] for i in range(N)]
        new_score = score
        for x, y in value:
            if not new_V[x][y]:
                new_V[x][y] = 1
                new_score += field[x][y]
        bt(value, new_score, new_V, time + 1)


N, M = map(int, input().split())

field = [list(map(int, input().split())) for _ in range(N)]
visited = [[0] * N for _ in range(N)]
my_loc = []
point = 0

for _ in range(M + 1):
    x, y = map(int, input().split())
    my_loc.append([x - 1, y - 1])
    point += field[x - 1][y - 1]
    visited[x - 1][y - 1] = 1

ans = point
bt(my_loc, point, visited, 0)

print(ans)
```

### [상미](./곰돌이의%20모험/상미.py)

```py

```

### [성구](./곰돌이의%20모험/성구.py)

```py
# 226ms 26MB
import sys
input = sys.stdin.readline

# 입력
N, M = map(int, input().split())
field = [[-1] * (N+1)]+[[-1] + list(map(int, input().split())) for _ in range(N)]
start = [0] * (M+1) # 시작 지점 점수 체크
teams = []
for i in range(M+1):
    team = tuple(map(int, input().split()))
    start[i] = field[team[0]][team[1]]
    field[team[0]][team[1]] = 0
    teams.append(team)

# 최대 점수, 초기값은 시작점 모두 더하기
limit = sum(start)
maxpoint = limit

def dfs(s:int, erazed:set, prev:int) -> None:
    global maxpoint

    if s > M:
        # 모두 체크했으면 최댓값 체크
        maxpoint = max(maxpoint, prev)
        return

    # 시작지점
    si, sj = teams[s]

    # 시작지점 체크

    stack = [(0, si, sj, start[s]+prev, set([(si, sj)]))]

    # 돌아다닐 필드 생성
    fields = [-1] * (N+1)
    for i in range(N+1):
        fields[i] = field[i].copy()

    # 몬스터 잡은 곳 체크
    for y, x in tuple(erazed):
        fields[y][x] = 0

    while stack:
        cnt, i, j, point, eraz = stack.pop()
        # 3시간 뒤 체크
        if cnt == 3:
            # 다음 사람 체크
            dfs(s+1, eraz, point)
            continue
        # 갈 수 있는 방향 모두 체크
        for di, dj in [(-1,0), (1,0) , (0,1), (0,-1), (0,0)]:
            ni, nj = di+i, dj+j
            if 0 < ni < N+1 and 0 < nj < N+1 and fields[ni][nj] >=0:
                # 움직였을 땐, 이동 경로 체크 및 점수 덧셈
                if (ni,nj) not in eraz:
                    e = eraz.copy()
                    e.add((ni,nj))
                    stack.append((cnt+1, ni, nj, point+fields[ni][nj], e))
                # 가만히 있었을 땐, 시간만 체크
                elif (i == ni and j == nj):
                    stack.append((cnt+1, ni, nj, point, eraz))
    return

dfs(0, set(), 0)

print(maxpoint)
```

### [승우](./곰돌이의%20모험/승우.py)

```py


```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

<img src="https://github.com/AlgoAlgo-ssafy-seoul-9th/25th_study/assets/102012985/4093d093-e9fb-49f5-885b-8fb00553c689 height="370">


</details>
