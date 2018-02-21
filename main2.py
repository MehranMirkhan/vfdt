
import time
import math
from multiprocessing import Process, Lock, Queue

import vfdt.dataset as ds
from vfdt import tree, measure


def train_model(dataset_info, config, train_df, test_df, accs, lock):
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
    with lock:
        model.show()
        print('Accuracy = {:.4f}'.format(acc))
        accs.put(acc)


if __name__ == '__main__':
    use_pool = True
    filepath = 'D:/data/rbf-a30-c6-n1e4.csv'
    n_instances = int(1e4)
    n_atts = 30
    n_classes = 6
    att_info = [('a{}'.format(i), 'numerical') for i in range(n_atts)]
    class_info = ['c{}'.format(i) for i in range(n_classes)]
    dataset_info = ds.DatasetInfo(att_info, class_info)
    delta = 1e-7
    R = math.log2(6)
    config = {
        'grace_period': 200,
        # 'impurity': measure.gini_index,
        'impurity': measure.information_entropy,
        # 'threshold': lambda N: measure.gini_quantile_bound(1e-1,
        #                                                    n_classes,
        #                                                    N),
        'threshold': measure.Hoeffding_bound_wrapper(R, delta),
        'tiebreak': 0.1,
        'min_var': 1e-12,
        'num_candids': 10
    }
    kfolds = 4
    fold_generator = ds.kfold_cross_validation(filepath, n_instances, kfolds)
    accs = Queue()
    procs = []
    lock = Lock()
    start = time.time()
    if use_pool:
        for train_df, test_df in fold_generator:
            p = Process(target=train_model,
                        args=(dataset_info, config,
                              train_df, test_df,
                              accs, lock))
            procs.append(p)
            p.start()
        for p in procs:
            p.join()
    else:
        for train_df, test_df in fold_generator:
            train_model(dataset_info, config, train_df, test_df, accs, lock)
    accs = [accs.get() for i in range(kfolds)]
    print('accuracy average: {}'.format(sum(accs)/len(accs)))
    print('--- {} seconds ---'.format(time.time() - start))
