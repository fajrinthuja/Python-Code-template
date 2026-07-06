import sys
import math
import bisect
from collections import namedtuple

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

def cmp(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0 

def solve(iterator):
    # C++: vector<int> a(n);        -> Python:
    n = int(next(iterator))
    m = int(next(iterator))
    a = [0] * n

    # C++: vector<vector<int>> a(n, vector<int>(m));  -> Python 2D:
    a = [[0] * m for _ in range(n)]   # NOT [[0]*m]*n -- that shares references!

    # reading n ints on one line:
    a = [int(next(iterator)) for _ in range(n)]

    # common ops
    x = 0
    a.append(x)      # push_back
    a.pop()           # pop_back
    a.pop(0)          # erase front (O(n), avoid in loops)
    a.sort()          # sort() -- in-place, ascending
    print(a)
    a.sort(reverse=True)   # sort() -- in-place, descending
    print(a)
    a = a[::-1]           # reversed copy
    len(a)            # size()
    print(a)

    # ---------- lower_bound / upper_bound (array must be sorted ascending) ----------
    a.sort()
    target = a[0] if a else 0
    lb = bisect.bisect_left(a, target)     # first index a[i] >= target
    ub = bisect.bisect_right(a, target)    # first index a[i] >  target
    print("lower_bound:", lb, "upper_bound:", ub)

    # ---------- custom comparator (sort by key function) ----------
    # e.g. sort by absolute value, ties broken by original value ascending
    a.sort(key=cmp)
    print("custom sort:", a)

        # --- vector<pair<int,int>> -> Python: list of tuples ---
    pairs = [(1, 5), (3, 2), (2, 8)]      # O(n) creation

    # reading n pairs from input:
    n = 3
    # pairs = [tuple(map(int, next(iterator) for _ in range(2))) for _ in range(n)]
    # with your token_generator (cleaner):
    pairs = [(int(next(iterator)), int(next(iterator))) for _ in range(n)]

    # sort by first element (default tuple comparison, like pair's default <):
    pairs.sort()                          # O(n log n) -- compares elem[0] then elem[1]

    # sort by second element (custom comparator):
    pairs.sort(key=lambda p: p[1])        # O(n log n)

    # sort by second descending, then first ascending:
    pairs.sort(key=lambda p: (-p[1], p[0]))   # O(n log n)

    # access like pair.first / pair.second:
    a, b = pairs[0]           # unpacking, O(1)
    first = pairs[0][0]        # O(1)
    second = pairs[0][1]       # O(1)


    # --- vector<vector<int>> as "vector of list" when a group needs to MUTATE ---
    groups = [[1, 2], [3, 4, 5]]          # tuples are immutable, so use lists if you'll modify contents
    groups[0].append(99)                   # O(1) amortized -- works because inner is a list, not tuple
    # groups[0] = (1,2); groups[0].append(99) -> AttributeError, tuples have no append

    # --- named version (readability, like a lightweight struct) ---
    Point = namedtuple("Point", ["x", "y"])
    pts = [Point(1, 5), Point(3, 2)]
    pts.sort(key=lambda p: p.y)            # O(n log n) -- readable field access instead of p[1]
    print(pts[0].x, pts[0].y)               # O(1)

    # --- lower_bound / upper_bound on vector of pairs (sorted by first, then second) ---
    pairs.sort()
    target = (2, 0)
    lb = bisect.bisect_left(pairs, target)   # O(log n) -- works because tuples compare lexicographically

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