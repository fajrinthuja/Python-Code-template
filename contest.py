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
    one = 0
    other = 0
    i = 0
    while i < n:
        if a[i] == 1:
            one += 1
        else:
            other += 1
        if one == other:
            i += 1
            break
        if one > other:
            if (i + 1) < n and a[i+1] == 3:
                i += 2
            else:
                i += 1
            break


        i += 1
    if i == n:
        print("NO")
        return
    one = 0
    other = 0
    # print(i)
    while i < n:
        if a[i] != 3:
            one += 1
        else:
            other += 1
        if one >= other:
            i += 1
            break
        i += 1
    if i < n:
        print("YES")
    else: 
        print("NO")
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