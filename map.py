import sys
import math
from sortedcontainers import SortedDict  # pip install sortedcontainers
# SortedDict = closest Python equivalent to C++ std::map (ordered, tree-based)
# Plain dict = closest equivalent to C++ std::unordered_map (hash-based)

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
    # ======================================================
    # 1) CREATING MAPS WITH DIFFERENT KEY TYPES
    # ======================================================

    # ---- int keys ----  (C++: map<int, V>)
    m_int = SortedDict()
    m_int[5] = "five"
    m_int[1] = "one"
    m_int[3] = "three"

    # ---- string keys ----  (C++: map<string, V>)
    m_str = SortedDict()
    m_str["banana"] = 3
    m_str["apple"] = 5

    # ---- pair keys ----  (C++: map<pair<int,int>, V>)
    # Python has no native "pair" -> use a tuple (int, int), which is hashable
    m_pair = SortedDict()
    m_pair[(1, 2)] = "point A"
    m_pair[(3, 4)] = "point B"

    # ---- vector keys ----  (C++: map<vector<int>, V>)
    # Python lists are NOT hashable -> must convert vector to tuple to use as key
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    m_vec = SortedDict()
    m_vec[tuple(vec1)] = "vector one"
    m_vec[tuple(vec2)] = "vector two"

    # ---- tuple of mixed types as key ----  (C++: map<tuple<int,string>, V>)
    m_mixed_key = SortedDict()
    v = [34,55]
    m_mixed_key[(1, "a", tuple(v))] = "mixed A"
    m_mixed_key[(2, "b", tuple(v))] = "mixed B"

    print(m_mixed_key)
    print("value : " + str(m_mixed_key[(1, 'a', tuple(v))]))

    # ======================================================
    # 2) STORING DIFFERENT VALUE TYPES
    # ======================================================

    m_val_int = SortedDict()
    m_val_int["count"] = 42                       # int value

    m_val_str = SortedDict()
    m_val_str["name"] = "Alice"                   # string value

    m_val_vector = SortedDict()
    m_val_vector["nums"] = [1, 2, 3]               # vector<int> value -> Python list

    m_val_pair = SortedDict()
    m_val_pair["coord"] = (10, 20)                 # pair<int,int> value -> tuple

    m_val_set = SortedDict()
    m_val_set["tags"] = {"red", "blue"}            # set<T> value -> Python set

    m_val_map = SortedDict()
    m_val_map["nested"] = SortedDict({"x": 1})     # map<K, map<K2,V2>> value -> nested SortedDict

    m_val_struct = SortedDict()
    m_val_struct["person"] = {"age": 30, "city": "NYC"}  # struct-like value -> dict

    # ======================================================
    # 3) EVERY COMMON C++ std::map FUNCTION, WITH PYTHON EQUIVALENT
    # ======================================================
    m = SortedDict()

    # --- insert / operator[] ---
    m["apple"] = 5              # m["apple"] = 5;                -> O(log n)
    m["banana"] = 3              # m.insert({"banana", 3});       -> O(log n)
    m.setdefault("banana", 7)    # m.insert({"banana", 7}) (no overwrite if exists) -> O(log n)
    print(m["banana"])
    m["banana"] = 55
    print(m["banana"])
    # --- at() / operator[] read ---
    val = m["apple"]             # m.at("apple") or m["apple"]    -> O(log n)

    # --- find() / count() ---
    exists = "apple" in m        # m.find("apple") != m.end()     -> O(log n)
    cnt = m.count("apple") if hasattr(m, "count") else (1 if "apple" in m else 0)
    # SortedDict has no .count(); membership test above is the direct equivalent -> O(log n)

    # --- erase() ---
    if "banana" in m:
        del m["banana"]           # m.erase("banana");             -> O(log n)
    print("banana" in m)
    m["banana"] = 34
    if "banana" in m:
        m.pop("banana", None)
    print("banana" in m)
    # or: m.pop("banana", None)   # safe erase, no KeyError if missing

    # --- size() / empty() ---
    size = len(m)                 # m.size()                       -> O(1)
    is_empty = len(m) == 0         # m.empty()                      -> O(1)

    # --- clear() ---
    m_copy = SortedDict(m)
    m_copy.clear()                 # m.clear();                     -> O(n)

    # --- begin()/rbegin() equivalents: smallest / largest key ---
    if len(m) > 0:
        first_key, first_val = m.peekitem(0)   # *m.begin()          -> O(log n)
        last_key, last_val = m.peekitem(-1)     # *m.rbegin()         -> O(log n)

    # --- iteration (in-order, like C++ map's sorted iteration) ---
    for k, v in m.items():        # for (auto& [k, v] : m)          -> O(n)
        print(str(k) + " " + str(v))
        pass

    # --- lower_bound() / upper_bound() ---
    # m.lower_bound(x): first key >= x   |  m.upper_bound(x): first key > x
    lb_index = m.bisect_left("b")           # index of lower_bound -> O(log n)
    ub_index = m.bisect_right("b")          # index of upper_bound -> O(log n)
    if lb_index < len(m):
        lower_bound_key = m.keys()[lb_index]   # -> O(log n) to fetch by index
        print(lower_bound_key)
    if ub_index < len(m):
        upper_bound_key = m.keys()[ub_index]   # -> O(log n)

    # --- keys() / values() ---
    all_keys = list(m.keys())      # for(auto& [k,_] : m) collect k -> O(n)
    all_values = list(m.values())  # for(auto& [_,v] : m) collect v -> O(n)

    # --- swap() ---
    m2 = SortedDict()
    m2["x"] = 1
    m, m2 = m2, m                  # m.swap(m2);                    -> O(1)

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