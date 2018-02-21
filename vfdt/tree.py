
import logging
from sklearn.base import BaseEstimator, ClassifierMixin

from . import suffstat
from . import util
from . import dataset as ds


class Node(object):
    def set_parent(self, parent):
        self.parent = parent


class LeafNode(Node):
    def __init__(self, dataset_info, available_atts, min_var, num_candids):
        """
        Args:
            dataset_info (DatasetInfo)
            available_atts (list): [True if att is available; False otherwise]
        Returns:
            object
        """
        self.dataset_info = dataset_info
        self.available_atts = available_atts
        self.suff_stats = {}
        self.class_counts = {}
        self.num_instances = 0              # Number of arrived instances
        for att_index, att_info in enumerate(dataset_info.att_info):
            att_name, values = att_info
            if available_atts[att_index]:
                if type(values) is str and values.lower() == "numerical":
                    ss = suffstat.SuffStatAttGaussian(min_var, num_candids)
                elif type(values) is list:
                    ss = suffstat.SuffStatAttDict(values)
                else:
                    raise(Exception("Wrong attribute: {}".format(values)))
                self.suff_stats[att_name] = ss

    def add_instance(self, instance, label):
        for att_index, att_info in enumerate(self.dataset_info.att_info):
            att_name, _ = att_info
            if self.available_atts[att_index]:
                self.suff_stats[att_name].add_value(instance[att_index],
                                                    label)
        if label in self.class_counts:
            self.class_counts[label] += 1
        else:
            self.class_counts[label] = 0
        self.num_instances += 1

    def check_split(self, impurity, threshold, tiebreak=0):
        """ Checks whether split is required.

        Args:
            impurity (function: list -> float): Calculates impurity of a list.
            threshold (function: int -> float): Calculates Hoeffding bound.
            tiebreak (float): Minimum allowed Hoeffding bound (default 0).

        Returns:
            None if split is not required;
            otherwise, split info.
        """
        class_counts = self.class_counts.values()       # ATTENTION: order is not preserved
        im = impurity(class_counts)          # impurity of this node
        N = self.num_instances
        best_gains = []
        for att_name, att_values in self.dataset_info.att_info:
            if att_name in self.suff_stats:
                ss = self.suff_stats[att_name]
                best_gains.append(im - ss.get_split_gain(impurity) / N)
            else:
                best_gains.append(-float('inf'))
        (a1_index, a1_gain), (a2_index, a2_gain) = util.get_top_two(best_gains)
        e = threshold(N)
        logging.debug('SplitCheck [a1 {}][a2 {}][e {}]'.format((a1_index,
                                                                a1_gain),
                                                               (a2_index,
                                                                a2_gain),
                                                               e))
        if a1_gain - a2_gain > e or e < tiebreak:      # Split
            att_name, att_values = self.dataset_info.att_info[a1_index]
            if type(att_values) is str and att_values.lower() == 'numerical':
                att_value = self.suff_stats[att_name].get_best_split_point()
                return a1_index, att_value
            else:
                raise(Exception("Splitting nominal attribute is not implemented yet."))
        else:
            return None

    def get_major_label(self):
        major_label = None
        major_count = 0
        if not self.class_counts:       # No data has arrived
            return None
        for label, count in self.class_counts.items():
            if count > major_count:
                major_label = label
                major_count = count
        return major_label

    def show(self, level):
        print('|  '*level, 'class {}'.format(self.get_major_label()))


class DecisionNodeNumerical(Node):
    def __init__(self, dataset_info, attribute_index, decision_value,
                 left_child, right_child):
        self.dataset_info = dataset_info
        self.attribute_index = attribute_index
        self.decision_value = decision_value
        self.left_child = left_child
        self.right_child = right_child

    def sort_down(self, instance):
        att_value = instance[self.attribute_index]
        if att_value >= self.decision_value:
            return self.right_child
        else:
            return self.left_child

    def replace_child(self, child, new_child):
        if child == self.left_child:
            self.left_child = new_child
        elif child == self.right_child:
            self.right_child = new_child
        else:
            raise(Exception("No such child."))

    def show(self, level):
        att_name = self.dataset_info.att_info[self.attribute_index][0]
        print('|  '*level,
              '{} >= {:.4f}'.format(att_name, self.decision_value))
        self.right_child.show(level + 1)
        print('|  '*level,
              '{} < {:.4f}'.format(att_name, self.decision_value))
        self.left_child.show(level + 1)


class DecisionNodeNominal(Node):
    def __init__(self, dataset_info, attribute_index, value_child_dict):
        self.dataset_info = dataset_info
        self.attribute_index = attribute_index
        self.value_child_dict = value_child_dict

    def sort_down(self, instance):
        att_value = instance[self.attribute_index]
        return self.value_child_dict[att_value]

    def replace_child(self, child, new_child):
        for _att_val, _child in self.value_child_dict.items():
            if _child == child:
                att_val = _att_val
                break
        self.value_child_dict[att_val] = new_child

    def show(self, level):
        att_name = self.dataset_info.att_info[self.attribute_index][0]
        for att_value, child in self.value_child_dict.items():
            print('|  '*level, '{} = {}'.format(att_name, att_value))
            self.child.show(level + 1)


class VFDT(object):
    """Very Fast Decision Tree
    """
    def __init__(self, dataset_info, config):
        self.dataset_info = dataset_info
        self.config = config
        available_atts = [True] * dataset_info.num_atts
        self.root = LeafNode(dataset_info, available_atts,
                             config['min_var'], config['num_candids'])
        self.root.set_parent(None)

    def sort_down(self, instance):
        node = self.root
        while(type(node) is not LeafNode):
            node = node.sort_down(instance)
        return node

    def learn(self, instance, label):
        leaf = self.sort_down(instance)
        leaf.add_instance(instance, label)
        if leaf.num_instances % self.config['grace_period'] == 0:
            result = leaf.check_split(self.config['impurity'],
                                      self.config['threshold'],
                                      self.config['tiebreak'])
            if result is not None:      # Split is required
                att_index, split_info = result
                att_name, values = self.dataset_info.att_info[att_index]
                if type(values) is str and values.lower() == "numerical":
                    left_child = LeafNode(self.dataset_info,
                                          leaf.available_atts,
                                          self.config['min_var'],
                                          self.config['num_candids'])
                    right_child = LeafNode(self.dataset_info,
                                           leaf.available_atts,
                                           self.config['min_var'],
                                           self.config['num_candids'])
                    dnode = DecisionNodeNumerical(self.dataset_info,
                                                  att_index, split_info,
                                                  left_child, right_child)
                    left_child.set_parent(dnode)
                    right_child.set_parent(dnode)
                    dnode.set_parent(leaf.parent)
                elif type(values) is list:
                    available_atts = list(leaf.available_atts)
                    available_atts[att_index] = False
                    value_child_dict = {}
                    for v in values:
                        value_child_dict[v] = LeafNode(self.dataset_info,
                                                       available_atts,
                                                       self.config['min_var'],
                                                       self.config['num_candids'])
                    dnode = DecisionNodeNominal(self.dataset_info,
                                                att_index,
                                                value_child_dict)
                    for node in value_child_dict.values():
                        node.set_parent(dnode)
                    dnode.set_parent(leaf.parent)
                else:
                    raise(Exception("Wrong attribute: {}".format(values)))
                if leaf.parent is not None:
                    leaf.parent.replace_child(leaf, dnode)
                else:
                    self.root = dnode

    def classify(self, instance):
        leaf = self.sort_down(instance)
        return leaf.get_major_label()

    def show(self):
        self.root.show(0)


# class VFDTClassifier(BaseEstimator, ClassifierMixin):
#     """VFDT classifier"""

#     def __init__(self, dataset_info, config):
#         """
#         Called when initializing the classifier
#         """
#         self.dataset_info = dataset_info
#         self.config = config
#         self.vfdt = VFDT(dataset_info, config)

#     def fit(self, X, y=None):
#         """
#         This should fit classifier. All the "work" should be done here.
#         """
#         for instance, label in ds.data_frame_iterator(X):
#             self.vfdt.learn(instance, label)
#         return self

#     def predict(self, X, y=None):
#         pred = []
#         for instance, label in ds.data_frame_iterator(X):
#             pred.append(self.vfdt.classify(instance))

#         return pred
