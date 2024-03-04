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