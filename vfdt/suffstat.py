""" Suffiecient Statistics
"""

import numpy as np
from scipy.stats import norm
from math import sqrt


class SuffStat(object):
    """ Sufficient Statistics for each attribute and class.

    Base class for inheritance.
    """
    def add_value(self, v):
        """ Adds a value to statistics.
        """

    def split(self):
        """Measures how many instances go into each partition
        """


class SuffStatDict(SuffStat):
    def __init__(self, values):
        self.values = values
        self.value_count = {}
        self.num_instances = 0

    def add_value(self, v):
        self.value_count[v] += 1
        self.num_instances += 1

    def split(self):
        """Measures how many instances go into each partition
        """
        raise("Not yet implemented.")


class SuffStatGaussian(SuffStat):
    def __init__(self, min_var=1e-12):
        self.sum = 0            # Sum of values
        self.sumsq = 0          # Sum of values squared
        self.num_instances = 0
        self.min_var = min_var

    def add_value(self, v):
        self.sum += v
        self.sumsq += v * v
        self.num_instances += 1

    def get_mean_var(self):
        if self.num_instances > 0:
            mean = self.sum / self.num_instances
            var = self.sumsq / self.num_instances - mean * mean
            if var < self.min_var:
                var = self.min_var
            return mean, var
        else:
            raise("No value has arrived yet.")

    def split(self, v):
        """Measures how many instances go into each partition
        """
        mean, var = self.get_mean_var()
        std = sqrt(var)
        n_l = self.num_instances * norm.cdf(v, loc=mean, scale=std)
        n_r = self.num_instances - n_l
        return (n_l, n_r)


class SuffStatAttDict(object):
    """ Sufficient Statistics for each attribute and all classes.

    Assuming the attribute is nominal.
    """
    def __init__(self, att_values, **kwargs):
        self.att_values = att_values
        self.stats = {}

    def add_value(self, v, label):
        if label in self.stats:
            self.stats[label].add_value(v)
        else:
            self.stats[label] = SuffStatDict(self.att_values)

    def get_split_gain(self):
        """ Measures information gain by splitting on this attribute.
        """
        raise("Not yet implemented.")


class SuffStatAttGaussian(object):
    """ Sufficient Statistics for each attribute and all classes.

    Assuming the attribute is numerical.
    """
    def __init__(self, num_candids=10, **kwargs):
        self.num_candids = num_candids
        self.stats = {}
        self.min_val = float('inf')
        self.max_val = -float('inf')

    def add_value(self, v, label):
        if v < self.min_val:
            self.min_val = v
        if v > self.max_val:
            self.max_val = v
        if label not in self.stats:
            self.stats[label] = SuffStatGaussian()
        self.stats[label].add_value(v)

    def get_split_gain(self, metric):
        """ Measures information gain by splitting on this attribute.

        Make sure some data has arrived already.

        Args:
            metric (function): An impurity measure function (like gini)

        Returns:
            best_im: Best impurity measure acheived by splitting with this attribute. ATTENTION: This is not the actual gain. gain = g(S) - best_im / N
            None: if check cannot be performed; There is only 1 class.
        """
        if len(self.stats.keys()) < 2:
            return None
        candid_points = np.linspace(self.min_val,
                                    self.max_val,
                                    self.num_candids + 2)
        candid_points = candid_points[1:-1]     # Disgarding first & last candidates
        # Computing [(N_l, N_r), ...] for each point
        class_counts_after_split = [[stat.split(v)
                                     for stat in self.stats.values()]
                                    for v in candid_points]
        # Computing [(N_l1, N_l2, ...), (N_r1, N_r2, ...)] for each point
        class_counts_after_split = [list(zip(*c))
                                    for c in class_counts_after_split]
        # Computing (g_l, g_r) and (n_l, n_r) for each point
        # im means impurity measure
        ims = [(metric(c[0]), metric(c[1])) for c in class_counts_after_split]
        sums = [(sum(c[0]), sum(c[1])) for c in class_counts_after_split]
        # Computing the best candidate point
        im = [s[0]*i[0] + s[1]*i[1] for i, s in zip(ims, sums)]
        best_im = min(im)
        best_im_index = im.index(best_im)
        best_point = candid_points[best_im_index]
        self.best_point = best_point
        return best_im
        # class_counts = [stat.num_instances for stat in self.stats]
        # im = metric(class_counts)           # impurity measure
        # N = sum(class_counts)
        # gain = im - best_im / N
        # return gain

    def get_best_split_point(self):
        """ Computes the best splitting point.
        """
        return self.best_point
