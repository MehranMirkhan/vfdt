
import unittest

from .. import tree


class Test_DecisionNode(unittest.TestCase):
    def test_nominal(self):
        c1 = "child1"
        c2 = "child2"
        c3 = "child3"
        node = tree.DecisionNodeNominal(1, {'a': c1, 'b': c2, 'c': c3})
        self.assertEqual(node.sort_down([-1, 'a', 3.3]), c1)
        self.assertEqual(node.sort_down([-1, 'b', 3.3]), c2)
        self.assertEqual(node.sort_down([-1, 'c', 3.3]), c3)

    def test_numeric(self):
        lc = "left_child"
        rc = "right_child"
        node = tree.DecisionNodeNumerical(1, 1.5, lc, rc)
        self.assertEqual(node.sort_down([-1, 0.9, 3.3]), lc)
        self.assertEqual(node.sort_down([-1, 1.9, 3.3]), rc)
