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
    try:
        n = int(next(iterator))
        
        # 1. SQUARE BRACKETS [] -> List (Ordered, mutable array)
        # Used here to grab the entire sequence of numbers
        # a = [int(next(iterator)) for _ in range(n)]
        # a = []
        # for i in range(n):
        #     x = int(next(iterator))
        #     a.append(x)
        a = [5] * n
        for i in range(n):
            a[i] = int(next(iterator))
        print(f"Original List: {a}")
        
        # 2. CURLY BRACES {} -> Set (Unique elements)
        # Great for filtering duplicates instantly
        unique_elements = set(a) 
        print(f"Unique Set: {unique_elements}")
        
        # 2b. CURLY BRACES {k: v} -> Dictionary (Key-Value pairs)
        # Let's count how many times each number appears
        frequency = {}
        for x in a:
            if x in frequency:
                frequency[x] += 1
            else:
                frequency[x] = 1
        print(f"Frequency Dictionary: {frequency}")
        
        # 3. PARENTHESES () -> Tuple (Immutable pair/group)
        # Let's find the minimum and maximum element and lock them in a tuple
        if a:
            min_max_pair = (min(a), max(a))
            print(f"Min and Max Tuple: {min_max_pair}")
            
        print("-" * 20) # Separator for test cases

    except StopIteration:
        return

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