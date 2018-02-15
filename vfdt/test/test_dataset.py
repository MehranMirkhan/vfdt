
import unittest
import pandas as pd

from .. import dataset


class Test_dataset(unittest.TestCase):
    def setUp(self):
        names = ['Bob', 'Jessica', 'Mary', 'John', 'Mel']
        births = [968, 155, 77, 578, 973]
        labels = ['c1', 'c2', 'c2', 'c1', 'c1']
        self.baby_dataset = list(zip(names, births, labels))
        self.df = pd.DataFrame(data=self.baby_dataset,
                               columns=['Names', 'Births', 'Class'])

    def test_iter(self):
        counter = 0
        for instance, label in dataset.data_frame_iterator(self.df):
            self.assertEqual(instance, self.baby_dataset[counter][:-1])
            self.assertEqual(label, self.baby_dataset[counter][-1])
            counter += 1

    def test_kfold_split(self):
        result = dataset.kfold_split(100, 4)
        result = list(result)
        result_expected = [
            (0, range(0, 25)),
            (25, range(25, 50)),
            (50, range(50, 75)),
            (75, range(75, 100))
        ]
        self.assertEqual(result, result_expected)

    def test_kfold_cv(self):
        filepath = './data/randomrbf.csv'
        num_instances = 100
        k = 10
        for train_df, test_df in dataset.kfold_cross_validation(filepath,
                                                                num_instances,
                                                                k):
            self.assertEqual(len(train_df), 90)
            self.assertEqual(len(test_df), 10)
