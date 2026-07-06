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
    # C++: vector<int> a(n);        -> Python:
    n = int(next(iterator))
    m = int(next(iterator))
    a = [0] * n

    # C++: vector<vector<int>> a(n, vector<int>(m));  -> Python 2D:
    a = [[0] * m for _ in range(n)]   # NOT [[0]*m]*n -- that shares references!

    # reading n ints on one line:
    # a = list(map(int, next(iterator) for _ in range(n)))
    # or, with your token_generator:
    a = [int(next(iterator)) for _ in range(n)]

    # common ops
    x = 0
    a.append(x)      # push_back
    a.pop()           # pop_back
    a.pop(0)          # erase front (O(n), avoid in loops)
    a.sort()          # sort() -- in-place
    print(a)
    a.sort(reverse=True)
    print(a)
    a = a[::-1]           # reversed copy
    len(a)            # size()
    print(a)

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