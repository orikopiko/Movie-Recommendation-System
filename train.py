from preprocess import Preprocessor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class Net(nn.Module):
    def __init__(self, n_hidden):
        super(Net, self).__init__()
        # print(n_hidden)
        self.fc1 = nn.Linear(n_hidden, 10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)
        # self.relu2 = nn.ReLU()
        # self.fc3 = nn.Linear(50,10)
        # self.relu3 = nn.ReLU()
        # self.fc4 = nn.Linear(10, 1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x = self.relu2(x)
        # x = self.fc3(x)
        # x = self.relu3(x)
        # x = self.fc4(x)
        return x


class Train:
    def __init__(self, data):
        self.data = data
        # print(data.columns)
        # print(data.info())
        # print(data)
        # print(data.shape[1])
        device = torch.device("cpu")
        net = Net(data.shape[1]-1).to(device) # our training data will have 1 less feature (drop rating)
        self.models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(),
            'Custom Neural Network': net
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
    def train_network(self, model, optimizer, criterion, batch_size = 64, n_epochs = 100):
        batch_start = torch.arange(0, self.X_tr.shape[0], batch_size)
        # Hold the best model
        best_mse = 100   # init to a number larger than the scale (0-5)
        best_weights = None
        history = []
        
        # for i in tqdm.tqdm(range(10)):
        #     print('hi')
        # training loop
        for epoch in range(n_epochs):
            model.train()
            with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
                bar.set_description(f"Epoch {epoch}")
                for start in bar:
                    # take a batch
                    start_index = start.item()
                    X_batch = self.X_tr[start_index:start_index+batch_size]
                    y_batch = self.y_tr[start_index:start_index+batch_size]
                    y_batch = torch.Tensor(y_batch.values)
                    optimizer.zero_grad()
                    

                    # forward pass
                    y_pred = model(torch.Tensor(X_batch.values))
                    loss = criterion(y_pred, y_batch)
                    # backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    # update weights
                    optimizer.step()
                    # print progress
                    bar.set_postfix(mse=float(loss))
            # evaluate accuracy at end of each epoch
            model.eval()
            y_pred = model(torch.Tensor(self.X_val.values))
            
            mse = criterion(y_pred, torch.Tensor(self.y_val.values))
            mse = float(mse)
            history.append(mse)
            if mse < best_mse:
                best_mse = mse
                best_weights = model.state_dict()
        print(f'Number of constraints: {self.X_tr.shape[0]}')
        # print(f'Number of weights: {best_weights.shape}')
        print(f'Best Validation RMSE: {np.sqrt(best_mse)}' )
        # restore model and return best accuracy
        model.load_state_dict(best_weights)
        return model

    def fit_models(self):
        for model_name, model in self.models.items():
            print(f"Training with {model_name}")
            if model_name == 'Custom Neural Network':
                n_epochs = 100   # number of epochs to run
                batch_size = 64  # size of each batch
                optimizer = optim.SGD(model.parameters(), lr=0.01)
                criterion = nn.MSELoss()
                self.train_network(model, optimizer, criterion, batch_size = batch_size, n_epochs=n_epochs)                

            else:
                model.fit(self.X_tr, self.y_tr)
                y_pred = model.predict(self.X_val)
                print(f'Root Mean squared error: {mean_squared_error(self.y_val, y_pred, squared=False)}' )

    

