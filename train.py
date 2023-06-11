from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


class Train:
    def __init__(self, data):
        self.data = data
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regression': DecisionTreeRegressor()
        }
        self.X_tr = None
        self.y_tr = None
        self.X_val = None
        self.y_val = None
    def split(self, split_train_test = 0.8):
        grouped = self.data.groupby('userId')

        # Define a function to split a group into train and test
        def split_group(group, split_size = split_train_test):
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
        df_tr_temp = pd.concat(train_groups)
        train_grouped = df_tr_temp.groupby('userId')
        train_groups, validation_groups = zip(*train_grouped.apply(split_group))
        # Concatenate the train and test groups into DataFrames
        df_tr = pd.concat(train_groups)
        df_ts = pd.concat(test_groups)
        df_val = pd.concat(validation_groups)
        y_tr = df_tr['rating']
        X_tr = df_tr.drop('rating', axis=1)
        y_ts = df_ts['rating']
        X_ts = df_ts.drop('rating', axis = 1)
        y_val = df_val['rating']
        X_val = df_val.drop('rating', axis=1)
        self.X_tr = X_tr
        self.y_tr = y_tr
        self.X_val = X_val
        self.y_val = y_val
        return X_tr, X_ts, y_tr, y_ts
    
    def fit_models(self):
        for model_name, model in self.models.items():
            print(f"Training with {model_name}")
            model.fit(self.X_tr, self.y_tr)
            y_pred = model.predict(self.X_val)
            print(f'Root Mean squared error: {mean_squared_error(self.y_val, y_pred, squared=False)}' )

    

