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
    # try:
        n = int(next(iterator))
        a = [int(next(iterator)) for _ in range(n)]
        print(a)
    # except StopIteration:
    #     return

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