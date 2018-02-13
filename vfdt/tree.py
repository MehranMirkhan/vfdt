

class LeafNode(object):
    def __init__(self):
        self.stat = None


class DecisionNode(object):
    def __init__(self):
        self.attribute_index = None
        self.left_child = None
        self.right_child = None


class BDTree(object):
    """
        Binary Decision Tree
    """
    def __init__(self):
        self.root = LeafNode()
