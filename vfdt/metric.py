""" Impurity measures
"""

def gini(n):
    s = sum(n)
    if s <= 0:
        raise("Gini index received bad vector.")
    p = [i/s for i in n]
    return 1 - sum([i*i for i in p])
