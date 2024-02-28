# 25th_study

<br/>

# 이번주 스터디 문제

<details markdown="1" open>
<summary>접기/펼치기</summary>

<br/>

## [두 용액](https://www.acmicpc.net/problem/2470)

### [민웅](./두%20용액/민웅.py)

```py

```

### [상미](./두%20용액/상미.py)

```py

```

### [성구](./두%20용액/성구.py)

```py

```

### [승우](./두%20용액/승우.py)

```py


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

```

### [승우](./곰돌이의%20모험/승우.py)

```py


```

</details>

<br/><br/>

# 알고리즘 설명

<details markdown="1">
<summary>접기/펼치기</summary>

</details>
