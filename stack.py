import sys
import math
from collections import deque  # deque = C++ stack equivalent (O(1) push/pop from one end)

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
    # ---- Python list as stack (simplest, most common choice) ----
    stack = []

    stack.append(5)              # st.push(5);                 -> O(1) amortized
    stack.append(1)              # st.push(1);                 -> O(1) amortized
    stack.append(3)              # st.push(3);                 -> O(1) amortized

    top = stack[-1]              # st.top();                   -> O(1)

    stack.pop()                  # st.pop();                   -> O(1) amortized

    size = len(stack)            # st.size();                  -> O(1)
    is_empty = len(stack) == 0    # st.empty();                 -> O(1)

    for x in reversed(stack):    # iterate top -> bottom (no direct C++ equiv, st has no iterator) -> O(n)
        pass

    # ---- different element types (any hashable or unhashable object works, unlike set/map) ----
    stack_int = []
    stack_int.append(10)                       # stack<int>

    stack_str = []
    stack_str.append("hello")                  # stack<string>

    stack_pair = []
    stack_pair.append((1, 2))                   # stack<pair<int,int>>

    stack_vec = []
    stack_vec.append([1, 2, 3])                  # stack<vector<int>> -> list works fine (no hashing needed)

    stack_struct = []
    stack_struct.append({"x": 1, "y": 2})         # stack of struct-like dict

    # ---- deque-based stack (use when you also need push/pop from the front) ----
    dstack = deque()
    dstack.append(1)              # push_back              -> O(1)
    dstack.append(2)              # push_back              -> O(1)
    top_d = dstack[-1]            # top                     -> O(1)
    dstack.pop()                  # pop_back                -> O(1)

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