
import pandas as pd


class DatasetInfo(object):
    def __init__(self, att_info, class_info):
        """
        Args:
            att_info (list): [(att_name, 'numerical' or [list of values]), ...]
            class_info (list): list of labels (e.g. ['c1', 'c2', ...])
        Returns:
            object
        """
        self.att_info = att_info
        self.num_atts = len(att_info)
        self.class_info = class_info
        self.num_classes = len(class_info)


def data_frame_iterator(data_frame, class_index=-1):
    """Iterates a pandas dataframe.

    Args:
        data_frame (pandas.DataFrame)
        class_index (int): the index of label column

    Yields:
        instance (tuple), label
    """
    for row in data_frame.itertuples():
        row = row[1:]
        label = row[class_index]
        if class_index == -1:
            instance = row[:-1]
        else:
            instance = row[0:class_index] + row[class_index + 1:]
        yield instance, label


def kfold_split(num_instances, k):
    """Generates indices for splitting a dataset.

    Example:
        kfold_idx = kfold_split(num_instances, k)
        for skip_row, skip_row_index in kfold_idx:
            test_df = pandas.read_csv(..., skiprows=skip_row, nrows=fold_size)
            train_df = pandas.read_csv(..., skiprows=skip_row_index)
    """
    fold_size = num_instances // k
    skip_row = [i * fold_size for i in range(k)]
    skip_row_index = [range(i, i+fold_size) for i in skip_row]
    return zip(skip_row, skip_row_index)


def kfold_cross_validation(filepath, num_instances, k, **kwargs):
    fold_size = num_instances // k
    kfold_idx = kfold_split(num_instances, k)
    for skip_row, skip_row_index in kfold_idx:
        test_df = pd.read_csv(filepath, skiprows=skip_row, nrows=fold_size,
                              **kwargs)
        train_df = pd.read_csv(filepath, skiprows=skip_row_index,
                               **kwargs)
        yield train_df, test_df
