
import math
from scipy.stats import norm

from . import util

"""Impurity measures
"""

def information_entropy(n):
    p = util.n2p(n)
    return -sum([i*math.log(i) for i in p])

def gini_index(n):
    p = util.n2p(n)
    return 1 - sum([i*i for i in p])

def misclassification_error(n):
    p = util.n2p(n)
    return 1 - max(p)


"""Threshold measures
"""

def Hoeffding_bound(R, delta, N):
    return R * math.sqrt(math.log(1/delta) / (2*N))

def misclassification_quantile_bound(delta, N):
    return norm.ppf(1-delta) / math.sqrt(2 * N)

def gini_quantile_bound(delta, K, N):
    return norm.ppf(1-delta) * math.sqrt((10*K*K - 16*K + 8) / N)
