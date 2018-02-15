
import pandas as pd

import vfdt.dataset as ds
from vfdt import tree, measure


def f1():
    file_path = './data/randomrbf.csv'
    df = pd.read_csv(file_path, nrows=4, skiprows=4)
    print(df)
    for instance, label in ds.data_frame_iterator(df):
        print('instance: {} - label: {}'.format(instance, label))


def f2():
    a = [[(2, 4), (1, 6)], [(5, 3), (3, 2)], [(6, 1), (4, 3)]]
    b = [list(zip(*q)) for q in a]
    print(b)
    c = [(sum(q[0]), sum(q[1])) for q in b]
    print(c)


def f3():
    filepath = './data/randomrbf.csv'
    num_instances = 100
    att_info = [('a{}'.format(i), 'numerical') for i in range(10)]
    class_info = ['c1', 'c2']
    num_classes = len(class_info)
    dataset_info = ds.DatasetInfo(att_info, class_info)
    config = {
        'grace_period': 10,
        'impurity': measure.gini_index,
        'threshold': lambda N: measure.gini_quantile_bound(0.05,
                                                           num_classes,
                                                           N),
        'tiebreak': 1e-2,
        'min_var': 1e-12,
        'num_candids': 10
    }
    model = tree.VFDT(dataset_info, config)
    for train_df, test_df in ds.kfold_cross_validation(filepath,
                                                       num_instances,
                                                       2):
        for instance, label in ds.data_frame_iterator(train_df):
            model.learn(instance, label)
        correct = 0
        num_tests = 0
        for instance, label in ds.data_frame_iterator(test_df):
            pred = model.classify(instance)
            if pred == label:
                correct += 1
            num_tests += 1
        print('Accuracy = {:.4f}'.format(correct / num_tests))


def main():
    f3()

if __name__ == '__main__':
    main()
