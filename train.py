from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Train:
    def __init__(self, data):
        self.data = data
    def split(self, split_size = 0.8):
        grouped = self.data.groupby('userId')

        # Define a function to split a group into train and test
        def split_group(group):
            train_size = int(split_size * len(group))
            indices = np.arange(len(group))
            np.random.shuffle(indices)
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            train_subset = group.take(train_indices)
            test_subset = group.take(test_indices)
            return train_subset, test_subset

        # Apply the split_group function to each group
        train_groups, test_groups = zip(*grouped.apply(split_group))

        # Concatenate the train and test groups into DataFrames
        train_df = pd.concat(train_groups)
        test_df = pd.concat(test_groups)
        
        return train_df, test_df
    
    def get_train_test(self):
        group = self.data.groupby('userId')
        # train_groups, test_groups = zip(*group.apply(split))

        # Concatenate the train and test groups into DataFrames
        # train_df = pd.concat(train_groups)
        # test_df = pd.concat(test_groups)
        # return train_df, test_df

    

