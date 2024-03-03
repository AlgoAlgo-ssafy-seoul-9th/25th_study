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