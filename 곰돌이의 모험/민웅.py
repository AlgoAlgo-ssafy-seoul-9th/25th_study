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