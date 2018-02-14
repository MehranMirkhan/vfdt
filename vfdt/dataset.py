
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


class DatasetCSV(object):
    def __init__(self, filepath, class_index=-1, **kwargs):
        """
        Args:
            filepath (str): The path to the dataset file.
            class_index (int): Index of class in each row.
        Returns:
            object
        """
        kwargs['chunksize'] = None
        self.data_frame = pd.read_csv(filepath, **kwargs)
        self.class_index = class_index

    def get_generator(self, epochs=1):
        for epoch in range(epochs):
            for row in self.data_frame.itertuples():
                row = row[1:]
                label = row[self.class_index]
                if self.class_index == -1:
                    instance = row[:-1]
                else:
                    instance = row[0:self.class_index] + row[self.class_index + 1:]
                yield instance, label


class DatasetCSVChunky(object):
    """Reads dataset in chunks. Useful when file is big and memory is short.
    """
    def __init__(self, filepath, chunksize, class_index=-1, **kwargs):
        """
        Args:
            filepath (str): The path to the dataset file.
            chunksize (int): Read the file 1 chunk at a time.
            class_index (int): Index of class in each row.
        Returns:
            object
        """
        self.data_generator = pd.read_csv(filepath,
                                          chunksize=chunksize,
                                          **kwargs)
        self.class_index = class_index

    def get_generator(self, epochs=1):
        for epoch in range(epochs):
            counter = 0
            for chunk in self.data_generator:
                print(counter)
                counter += 1
                for row in chunk.itertuples():
                    row = row[1:]
                    label = row[self.class_index]
                    if self.class_index == -1:
                        instance = row[:-1]
                    else:
                        instance = row[0:self.class_index] + row[self.class_index + 1:]
                    yield instance, label
