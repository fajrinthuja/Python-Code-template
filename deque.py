import sys
import math
from collections import deque  # deque = C++ std::deque equivalent (O(1) push/pop at both ends)

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
    dq = deque()

    # ---- push / pop from both ends ----
    dq.append(5)                # dq.push_back(5);            -> O(1)
    dq.appendleft(1)             # dq.push_front(1);            -> O(1)
    dq.append(3)                 # dq.push_back(3);             -> O(1)

    back = dq[-1]                 # dq.back();                   -> O(1)
    front = dq[0]                  # dq.front();                  -> O(1)

    dq.pop()                       # dq.pop_back();                -> O(1)
    dq.popleft()                    # dq.pop_front();                -> O(1)

    size = len(dq)                   # dq.size();                     -> O(1)
    is_empty = len(dq) == 0           # dq.empty();                     -> O(1)

    # ---- random access (deque supports indexing, unlike stack/queue) ----
    dq2 = deque([10, 20, 30, 40])
    val = dq2[2]                        # dq[2];                          -> O(1)
    dq2[2] = 99                          # dq[2] = 99;                      -> O(1)

    # ---- insert / erase at arbitrary position ----
    dq2.insert(1, 15)                     # dq.insert(dq.begin()+1, 15);     -> O(n)
    del dq2[1]                             # dq.erase(dq.begin()+1);          -> O(n)

    # ---- clear ----
    dq2.clear()                             # dq.clear();                       -> O(n)

    # ---- iterate ----
    for x in dq:                             # for (auto& x : dq)                -> O(n)
        pass
    for x in reversed(dq):                    # for (auto it = dq.rbegin(); ...) -> O(n)
        pass

    # ---- rotate (useful for sliding-window / circular problems) ----
    dq3 = deque([1, 2, 3, 4, 5])
    dq3.rotate(1)                              # shifts right: [5,1,2,3,4]         -> O(k)
    dq3.rotate(-1)                               # shifts left                       -> O(k)

    # ---- different element types (no hashing needed) ----
    dq_str = deque(["a", "b"])                    # deque<string>
    dq_pair = deque([(1, 2), (3, 4)])              # deque<pair<int,int>>
    dq_vec = deque([[1, 2], [3, 4]])                 # deque<vector<int>>

    # ---- bounded deque (auto-discards from opposite end when full, e.g. sliding window) ----
    window = deque(maxlen=3)
    for x in [1, 2, 3, 4, 5]:
        window.append(x)                             # old front auto-popped once full -> O(1)

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