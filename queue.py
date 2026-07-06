import sys
import math
from collections import deque  # deque = C++ queue equivalent (O(1) push/pop from both ends)

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
    # ======================================================
    # QUEUE TEMPLATE  (q = deque())
    # ======================================================
    q = deque()

    q.append(5)                 # q.push(5);                  -> O(1)
    q.append(1)                 # q.push(1);                  -> O(1)
    q.append(3)                 # q.push(3);                  -> O(1)

    front = q[0]                 # q.front();                  -> O(1)
    back = q[-1]                  # q.back();                   -> O(1)

    q.popleft()                  # q.pop();                    -> O(1)

    size = len(q)                 # q.size();                   -> O(1)
    is_empty = len(q) == 0         # q.empty();                  -> O(1)

    for x in q:                   # iterate front -> back (no direct C++ equiv) -> O(n)
        pass

    # different element types (no hashing needed, unlike set/map)
    q_str = deque(["a", "b"])       # queue<string>
    q_pair = deque([(1, 2), (3, 4)]) # queue<pair<int,int>>
    q_vec = deque([[1, 2], [3, 4]])   # queue<vector<int>>

    # ======================================================
    # BFS USING THE QUEUE + ADJACENCY LIST (vector<vector<int>>)
    # ======================================================
    print("enter number of nodes : ")
    n = int(next(iterator))          # number of nodes
    print("enter number of edges : ")
    m = int(next(iterator))          # number of edges
    print("Enter edges : ")

    adj = [[] for _ in range(n + 1)] # vector<vector<int>> adj(n+1);  -> O(n)

    for _ in range(m):
        u = int(next(iterator))
        v = int(next(iterator))
        adj[u].append(v)             # adj[u].push_back(v);         -> O(1) amortized
        adj[v].append(u)             # adj[v].push_back(u); (undirected) -> O(1) amortized

    print("enter src : ")
    src = int(next(iterator))

    dist = [-1] * (n + 1)            # vector<int> dist(n+1, -1);    -> O(n)
    dist[src] = 0

    bfs_q = deque()             # queue<int> q; q.push(src);   -> O(1)
    bfs_q.append(src)

    while bfs_q:                     # while (!q.empty())            -> O(V)
        u = bfs_q.popleft()          # int u = q.front(); q.pop();   -> O(1)
        for v in adj[u]:             # for (int v : adj[u])          -> O(E) total across all u
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                bfs_q.append(v)      # q.push(v);                    -> O(1)

    print(dist[1:])                  # distances from src to every node (1-indexed)

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