
import unittest

from .. import measure


class Test_gini(unittest.TestCase):
    def test_corrrectness(self):
        n = [2, 1, 3, 4]
        im = measure.gini_index(n)
        expected = 0.7
        self.assertAlmostEqual(im, expected)

class Test_Bounds(unittest.TestCase):
    def test_mis_quan_bound(self):
        bound = measure.misclassification_quantile_bound(0.05, 1e4)
        expect = 0.01163087
        self.assertAlmostEqual(bound, expect)
