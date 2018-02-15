
import unittest

from .. import tree
from .. import dataset
from .. import measure


class Test_DecisionNode(unittest.TestCase):
    def test_nominal(self):
        c1 = "child1"
        c2 = "child2"
        c3 = "child3"
        node = tree.DecisionNodeNominal(1, {'a': c1, 'b': c2, 'c': c3})
        self.assertEqual(node.sort_down([-1, 'a', 3.3]), c1)
        self.assertEqual(node.sort_down([-1, 'b', 3.3]), c2)
        self.assertEqual(node.sort_down([-1, 'c', 3.3]), c3)

    def test_nominal2(self):
        c1 = "child1"
        c2 = "child2"
        c3 = "child3"
        node = tree.DecisionNodeNominal(1, {5: c1, 2: c2, -1: c3})
        self.assertEqual(node.sort_down([-1, 5, 3.3]), c1)
        self.assertEqual(node.sort_down([-1, 2, 3.3]), c2)
        self.assertEqual(node.sort_down([-1, -1, 3.3]), c3)

    def test_numeric(self):
        lc = "left_child"
        rc = "right_child"
        node = tree.DecisionNodeNumerical(1, 1.5, lc, rc)
        self.assertEqual(node.sort_down([-1, 0.9, 3.3]), lc)
        self.assertEqual(node.sort_down([-1, 1.9, 3.3]), rc)


class Test_LeafNode(unittest.TestCase):
    def setUp(self):
        att_info = [('a1', 'numerical'), ('a2', 'numerical')]
        available_atts = [True] * 2
        class_info = ['c1', 'c2']
        num_candids = 3
        self.dataset_info = dataset.DatasetInfo(att_info, class_info)
        self.node = tree.LeafNode(self.dataset_info, available_atts,
                                  min_var=1e-12,
                                  num_candids=num_candids)

    def test_split_check(self):
        g = measure.gini_index

        def th(N):
            return 0
        ds = [
            (-3, 1, 'c1'),
            (-1, -1, 'c1'),
            (1, -0.5, 'c2'),
            (3, 0.5, 'c2'),
            (-2, 0, 'c1')
        ]
        for d in ds:
            instance = d[:-1]
            label = d[-1]
            self.node.add_instance(instance, label)
        att_index, att_value = self.node.check_split(g, th)
        self.assertEqual(att_index, 0)
        self.assertEqual(att_value, 0)
        self.assertEqual(self.node.classify((-1, 0)), 'c1')
