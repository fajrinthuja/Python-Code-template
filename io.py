import sys
import math

sys.setrecursionlimit(300000)

inf = 10**18
M = 10**9 + 7
mod = 998244353
eps = 1e-7
pi = 2 * math.acos(0)

# Helper function to get individual tokens line-by-line
def token_generator():
    for line in sys.stdin:
        for token in line.split():
            yield token

def solve(iterator):
    n = int(next(iterator))
    a = [0] * n
    for i in range(n):
        a[i] = int(next(iterator))
    maxi = 0
    ans = 0
    for i in range (n):
        maxi = max(maxi,a[i])
        ans += maxi - a[i]
    print(ans)
    pass

def main():
    iterator = token_generator()
    
    try:
        t = int(next(iterator))
    except StopIteration:
        return
        
    for _ in range(t):
        solve(iterator)

if __name__ == '__main__':
    main()