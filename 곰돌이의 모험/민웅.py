import sys
from itertools import product

input = sys.stdin.readline
dxy = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 재귀탐색
def bt(loc, score, V, time):
    global ans
    # 3시간 다 돌았으면 return 및 정답갱신
    if time == 3:
        if score > ans:
            ans = score
        return

    # 각 좌표별로 갈수 있는 후보좌표들 저장할 리스트
    tmp_loc = [[] for _ in range(len(loc))]
    for i in range(len(loc)):
        x, y = loc[i]

        for d in dxy:
            nx = x + d[0]
            ny = y + d[1]
            # 벽 아니면 후보좌표 저장
            if 0 <= nx <= N - 1 and 0 <= ny <= N - 1:
                if field[nx][ny] != -1:
                    tmp_loc[i].append([nx, ny])
    # 시작부터 벽에 다 막혀있으면 좌표가 안나옴. 그래서 이동가능한 공간이 있는 좌표들 idx 체크하려고 만든리스트
    idx_check = []
    # 이동가능한 좌표 체크
    for i in range(len(loc)):
        if tmp_loc[i]:
            idx_check.append(i)
    idx_cnt = len(idx_check)
    # 이동 가능한 좌표에 있는 후보좌표들로 product 돌림
    # product 설명
    # 만약 test1 = [0, 1, 2], test2 = [3, 4] 일 때,

    # prod = product(test1, test2)
    # for p in prod:
    #     print(p)

    # (0, 3), (0, 4), (1, 3), (1, 4) ... 이렇게 나옴
    if idx_cnt == 3:
        prod = product(tmp_loc[idx_check[0]], tmp_loc[idx_check[1]], tmp_loc[idx_check[2]])
    elif idx_cnt == 2:
        prod = product(tmp_loc[idx_check[0]], tmp_loc[idx_check[1]])
    elif idx_cnt == 1:
        prod = product(tmp_loc[idx_check[0]])
    else:
        return

    # 좌표 후보군들 조합으로 이동하면서 점수이미 얻은 위치체크(visited 배열)하고 다음 재귀로 넘김
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