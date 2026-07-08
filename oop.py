import sys
import math
import functools

sys.setrecursionlimit(300000)

inf = 10**18
M = 10**9 + 7
mod = 998244353
eps = 1e-7
pi = 2 * math.acos(0)

# ---------------- Input (unchanged) ----------------
def token_generator():
    for line in sys.stdin:
        for token in line.split():
            yield token


# ---------------- OOP Template (like a struct/class in C++/Java) ----------------
class Employee:
    def __init__(self, id=0, name="", salary=0):
        self.id = id
        self.name = name
        self.salary = salary

    # ---- getters ----
    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_salary(self):
        return self.salary

    # ---- setters ----
    def set_id(self, id):
        self.id = id

    def set_name(self, name):
        self.name = name

    def set_salary(self, salary):
        self.salary = salary

    # ---- update (like a method that modifies state) ----
    def give_raise(self, amount):
        self.salary += amount

    # ---- string representation ----
    def __str__(self):
        return f"Employee(id={self.id}, name={self.name}, salary={self.salary})"

    def __repr__(self):
        return self.__str__()

    # ---- comparisons (for sorting) ----
    def __lt__(self, other):
        return self.salary < other.salary   # default: sort by salary

    def __eq__(self, other):
        return self.salary == other.salary


# ---------------- solve() using the class ----------------
def solve(iterator):
    print("Enter number of employees : ")
    n = int(next(iterator))
   
    employees = []
    for _ in range(n):
        print("enter id, name and salary : ")
        eid = int(next(iterator))
        name = next(iterator)
        salary = int(next(iterator))
        employees.append(Employee(eid, name, salary))

    # sort using default __lt__ (by salary)
    employees.sort()
    print("Sorted by salary:")
    for e in employees:
        print(e)

    # sort using a custom key (e.g., by name)
    employees.sort(key=lambda e: e.get_name())
    print("Sorted by name:")
    for e in employees:
        print(e)

    # sort using functools.cmp_to_key for multi-criteria custom comparator
    def cmp(a, b):
        if a.get_salary() != b.get_salary():
            return a.get_salary() - b.get_salary()
        return (a.get_name() > b.get_name()) - (a.get_name() < b.get_name())

    employees.sort(key=functools.cmp_to_key(cmp))
    print("Sorted by salary then name:")
    for e in employees:
        print(e)


def main():
    print("test cases : ")
    iterator = token_generator()
    try:
        t = int(next(iterator))
    except StopIteration:
        return
    for _ in range(t):
        solve(iterator)


if __name__ == '__main__':
    main()