
import pandas as pd


class Dataset(object):
    def __init__(self, att_types, num_classes):
        """
        Args:
            att_types (list): Type of attributes -> ['Nominal'/'Numeric']
            num_classes (int): Number of classes.
        Returns:
            object
        """
        self.att_types = att_types
        self.num_atts = len(att_types)
        self.num_classes = num_classes


class DatasetCSV(Dataset):
    def __init__(self, att_types, num_classes,
                 filepath, class_index=-1, **kwargs):
        """
        Args:
            att_types (list): Type of attributes -> ['Nominal'/'Numeric']
            num_classes (int): Number of classes.
            filepath (str): The path to the dataset file.
            class_index (int): Index of class in each row.
        Returns:
            object
        """
        Dataset.__init__(self, att_types, num_classes)
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


class DatasetCSVChunky(Dataset):
    """Reads dataset in chunks. Useful when file is big and memory is short.
    """
    def __init__(self, att_types, num_classes,
                 filepath, chunksize, class_index=-1, **kwargs):
        """
        Args:
            att_types (list): Type of attributes -> ['Nominal'/'Numeric']
            num_classes (int): Number of classes.
            filepath (str): The path to the dataset file.
            chunksize (int): Read the file 1 chunk at a time.
            class_index (int): Index of class in each row.
        Returns:
            object
        """
        Dataset.__init__(self, att_types, num_classes)
        self.data_generator = pd.read_csv(filepath,
                                          chunksize=chunksize,
                                          **kwargs)
        self.class_index = class_index

    def get_generator(self, epochs=1):
        for epoch in range(epochs):
            for chunk in self.data_generator:
                for row in chunk.itertuples():
                    row = row[1:]
                    label = row[self.class_index]
                    if self.class_index == -1:
                        instance = row[:-1]
                    else:
                        instance = row[0:self.class_index] + row[self.class_index + 1:]
                    yield instance, label
