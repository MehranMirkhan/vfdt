
import unittest

from .. import metric


class Test_gini(unittest.TestCase):
    def test_corrrectness(self):
        n = [2, 1, 3, 4]
        im = metric.gini(n)
        expected = 0.7
        self.assertAlmostEqual(im, expected)

    def test_exception(self):
        with self.assertRaises(Exception):
            metric.gini([0, 0, 0])
        with self.assertRaises(Exception):
            metric.gini([1, 0, -2])
