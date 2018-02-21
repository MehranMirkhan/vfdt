
import pandas as pd
import logging
import math
import time

import vfdt.dataset as ds
from vfdt import tree, measure


logging.basicConfig(level=logging.ERROR)


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
    accs = []
    for train_df, test_df in ds.kfold_cross_validation(filepath,
                                                       num_instances,
                                                       2):
        for instance, label in ds.data_frame_iterator(train_df):
            model.learn(instance, label)
        model.show()
        correct = 0
        num_tests = 0
        for instance, label in ds.data_frame_iterator(test_df):
            pred = model.classify(instance)
            if pred == label:
                correct += 1
            num_tests += 1
        acc = correct / num_tests
        print('Accuracy = {:.4f}'.format(acc))
        accs.append(acc)
    print('accuracy average: {}'.format(sum(accs)/len(accs)))


def f4():
    filepath = 'D:/data/rbf-a30-c6-n1e4.csv'
    num_instances = int(1e4)
    num_atts = 30
    num_classes = 6
    att_info = [('a{}'.format(i), 'numerical') for i in range(num_atts)]
    class_info = ['c{}'.format(i) for i in range(num_classes)]
    dataset_info = ds.DatasetInfo(att_info, class_info)
    delta = 1e-7
    R = math.log2(num_classes)
    config = {
        'grace_period': 200,
        # 'impurity': measure.gini_index,
        'impurity': measure.information_entropy,
        # 'threshold': lambda N: measure.gini_quantile_bound(1e-1,
        #                                                    num_classes,
        #                                                    N),
        'threshold': lambda N: measure.Hoeffding_bound(R, delta, N),
        'tiebreak': 0.1,
        'min_var': 1e-12,
        'num_candids': 10
    }
    kfolds = 4
    accs = []
    for train_df, test_df in ds.kfold_cross_validation(filepath,
                                                       num_instances,
                                                       kfolds):
        train_model(dataset_info, config, train_df, test_df, accs)
    print('accuracy average: {}'.format(sum(accs)/len(accs)))


def train_model(dataset_info, config, train_df, test_df, results):
    model = tree.VFDT(dataset_info, config)
    for instance, label in ds.data_frame_iterator(train_df):
        model.learn(instance, label)
    correct = 0
    num_tests = 0
    for instance, label in ds.data_frame_iterator(test_df):
        pred = model.classify(instance)
        if pred == label:
            correct += 1
        num_tests += 1
    acc = correct / num_tests
    model.show()
    print('Accuracy = {:.4f}'.format(acc))
    results.append(acc)


def f5():
    delta = 1e-7
    cc = [100, 5000, 4000, 100, 100, 100]
    K = len(cc)
    R = math.log2(K)
    print('gini-bound')
    print(measure.gini_quantile_bound(delta, K, 1e2))
    print(measure.gini_quantile_bound(delta, K, 1e3))
    print(measure.gini_quantile_bound(delta, K, 1e4))
    print('gini-index', measure.gini_index(cc))
    print('Hoeffding-bound')
    print(measure.Hoeffding_bound(R, delta, 1e2))
    print(measure.Hoeffding_bound(R, delta, 1e3))
    print(measure.Hoeffding_bound(R, delta, 1e4))
    print('entropy-index', measure.information_entropy(cc))
    print('mis-bound')
    print(measure.misclassification_quantile_bound(delta, 1e2))
    print(measure.misclassification_quantile_bound(delta, 1e3))
    print(measure.misclassification_quantile_bound(delta, 1e4))
    print('mis-index', measure.misclassification_error(cc))


def main():
    start = time.time()
    f4()
    print('--- {} seconds ---'.format(time.time() - start))

if __name__ == '__main__':
    main()
