
import unittest

from .. import util


class Test_top_two(unittest.TestCase):
    def test_correctness(self):
        vec = [-1, 3, 0, -7, 6, 12, 4, -2]
        (m1_idx, m1), (m2_idx, m2) = util.get_top_two(vec)
        self.assertEqual(m1_idx, 5)
        self.assertEqual(m1, 12)
        self.assertEqual(m2_idx, 4)
        self.assertEqual(m2, 6)


class Test_n2p(unittest.TestCase):
    def test_exception(self):
        with self.assertRaises(Exception):
            util.n2p([0, 0, 0])
        with self.assertRaises(Exception):
            util.n2p([1, 0, -2])
