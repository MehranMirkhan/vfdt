
import unittest

from . import stat


class Test_SuffStatGaussian(unittest.TestCase):
    def test_mean_var(self):
        s = stat.SuffStatGaussian()
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
        s = stat.SuffStatGaussian()
        s.add_value(1)
        s.add_value(2)
        s.add_value(3)
        s.add_value(4)
        n_l, n_r = s.split(2)
        cdf = 0.32736042300928847
        self.assertAlmostEqual(n_l, cdf*4)
        self.assertAlmostEqual(n_r, (1-cdf)*4)
