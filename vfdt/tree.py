
from . import suffstat
from . import util


class LeafNode(object):
    def __init__(self, dataset_info, available_atts):
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
        for att_index, att_info in enumerate(dataset_info.att_info):
            att_name, values = att_info
            if available_atts[att_index]:
                if type(values) is str and values.lower() == "numerical":
                    ss = suffstat.SuffStatAttGaussian(dataset_info.num_classes)
                elif type(values) is list:
                    ss = suffstat.SuffStatAttDict(values,
                                                  dataset_info.num_classes)
                else:
                    raise("Wrong attribute: {}".format(values))
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

    def check_split(self, metric, threshold, tie_break=0):
        """ Checks whether split is required.

        Args:
            metric (function: list -> float): Calculates impurity of a list.
            threshold (function: int -> float): Calculates Hoeffding bound.
            tie_break (float): Minimum allowed Hoeffding bound (default 0).

        Returns:
            None if split is not required;
            otherwise, split info.
        """
        class_counts = self.class_counts.values()       # ATTENTION: order is not preserved
        im = metric(class_counts)          # impurity of this node
        N = sum(class_counts)              # Number of arrived instances
        best_gains = [im - ss.get_split_gain(metric) / N
                      for ss in self.suff_stats]
        (a1_index, a1_gain), (a2_index, a2_gain) = util.get_top_two(best_gains)
        e = threshold(N)
        if a1_gain - a2_gain > e or e < tie_break:      # Split
            att_type = self.att_types[a1_index]
            if att_type == 'numerical':
                att_value = self.suff_stats(a1_index).get_best_split_point()
                return a1_index, att_value
            else:
                raise("Splitting nominal attribute is not implemented yet.")
        else:
            return None


class DecisionNodeNumerical(object):
    def __init__(self, attribute_index, decision_value,
                 left_child, right_child):
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


class DecisionNodeNominal(object):
    def __init__(self):
        raise("Not yet implemented.")


class VFDT(object):
    """Very Fast Decision Tree
    """
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.root = LeafNode()
