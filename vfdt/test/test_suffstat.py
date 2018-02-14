
import unittest

from .. import suffstat
from .. import metric


class Test_SuffStatGaussian(unittest.TestCase):
    def test_mean_var(self):
        s = suffstat.SuffStatGaussian()
        s.add_value(-1)
        s.add_value(1)
        mean, var = s.get_mean_var()
        self.assertAlmostEqual(mean, 0)
        self.assertAlmostEqual(var, 1)
        s.add_value(0)
        mean, var = s.get_mean_var()
        self.assertAlmostEqual(mean, 0)
        self.assertAlmostEqual(var, 2/3)

    def test_split(self):
        s = suffstat.SuffStatGaussian()
        s.add_value(1)
        s.add_value(2)
        s.add_value(3)
        s.add_value(4)
        n_l, n_r = s.split(2)
        cdf = 0.32736042300928847
        self.assertAlmostEqual(n_l, cdf*4)
        self.assertAlmostEqual(n_r, (1-cdf)*4)


class Test_SuffStatAttGaussian(unittest.TestCase):
    def test_correctness(self):
        g = metric.gini
        s = suffstat.SuffStatAttGaussian(num_candids=2)
        s.add_value(-2, 0)
        s.add_value(0, 0)
        s.add_value(1, 1)
        s.add_value(2, 1)
        s.add_value(3, 1)

        # Calculated by hand
        p1_im_expected = 0.8769368667209411
        p2_im_expected = 0.9848249911965317

        # Quick calculation
        k1 = s.stats[0]
        k2 = s.stats[1]
        p1, p2 = -1/3, 4/3              # Candidate points
        k1p1l, k1p1r = k1.split(p1)
        k1p2l, k1p2r = k1.split(p2)
        k2p1l, k2p1r = k2.split(p1)
        k2p2l, k2p2r = k2.split(p2)
        p1_im = (k1p1l + k2p1l) * g([k1p1l, k2p1l]) +\
                (k1p1r + k2p1r) * g([k1p1r, k2p1r])
        p2_im = (k1p2l + k2p2l) * g([k1p2l, k2p2l]) +\
                (k1p2r + k2p2r) * g([k1p2r, k2p2r])
        self.assertAlmostEqual(p1_im, p1_im_expected)
        self.assertAlmostEqual(p2_im, p2_im_expected)

        # Using source code
        best_im = s.get_split_gain(g)
        self.assertAlmostEqual(best_im, min(p1_im_expected, p2_im_expected))
