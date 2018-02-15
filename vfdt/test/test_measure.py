
import unittest

from .. import measure


class Test_gini(unittest.TestCase):
    def test_corrrectness(self):
        n = [2, 1, 3, 4]
        im = measure.gini_index(n)
        expected = 0.7
        self.assertAlmostEqual(im, expected)
