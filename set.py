import sys
import math
from sortedcontainers import SortedSet  # pip install sortedcontainers -> C++ std::set equivalent (ordered)

sys.setrecursionlimit(300000)

inf = 10**18
M = 10**9 + 7
mod = 998244353
eps = 1e-7
pi = 2 * math.acos(0)

def token_generator():
    for line in sys.stdin:
        for token in line.split():
            yield token

def solve(iterator):
    # ---- different element types (all must be hashable) ----
    s_int = SortedSet([3, 1, 2])          # set<int>
    s_str = SortedSet(["b", "a", "c"])    # set<string>
    s_pair = SortedSet([(2, 1), (1, 2)])  # set<pair<int,int>> -> tuples
    s_vec = SortedSet([tuple([1, 2]), tuple([0, 1])])  # set<vector<int>> -> list must become tuple
    print(s_vec)

    s = SortedSet()

    s.add(5)                     # s.insert(5);                -> O(log n)
    s.add(1)                     # s.insert(1);                -> O(log n)
    s.add(3)                     # s.insert(3);                -> O(log n)

    exists = 3 in s               # s.find(3) != s.end()        -> O(log n)

    if 1 in s:
        s.remove(1)                # s.erase(1);                  -> O(log n)
    s.discard(99)                  # safe erase, no error if missing -> O(log n)

    size = len(s)                  # s.size()                     -> O(1)
    is_empty = len(s) == 0          # s.empty()                    -> O(1)

    if len(s) > 0:
        smallest = s[0]             # *s.begin()                  -> O(log n)
        largest = s[-1]             # *s.rbegin()                 -> O(log n)

    for x in s:                    # for (auto& x : s)            -> O(n)
        pass

    lb_idx = s.bisect_left(3)       # s.lower_bound(3)             -> O(log n)
    ub_idx = s.bisect_right(3)      # s.upper_bound(3)              -> O(log n)
    if lb_idx < len(s):
        lower_bound_val = s[lb_idx]
    if ub_idx < len(s):
        upper_bound_val = s[ub_idx]

    s2 = SortedSet([3, 4, 5])
    union_ = s | s2                 # set union                    -> O(len(s)+len(s2))
    inter_ = s & s2                 # set intersection              -> O(min)
    diff_ = s - s2                  # set difference                -> O(len(s))

    s_copy = SortedSet(s)
    s_copy.clear()                  # s.clear();                   -> O(n)

    s, s2 = s2, s                   # s.swap(s2);                   -> O(1)

    # ---- C++ unordered_set / multiset equivalents ----
    unordered = set()               # unordered_set<int> -> O(1) avg insert/find/erase
    unordered.add(7)

    from collections import Counter
    multiset = Counter()            # multiset<int> -> counts duplicates, O(1) avg ops
    multiset[5] += 1
    multiset[5] += 1
    multiset[6] += 1
    mp = Counter()
    mp[(1,2)] = 55
    mp[(1,2)] += 2
    print(mp)
    print(multiset)

    print("done")

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