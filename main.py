from preprocess import Preprocessor
from train import Train
import numpy as np
import pandas as pd

def main():
    preprocessor = Preprocessor('datasets/ratings.csv', 'datasets/movies.csv', 'datasets/tags.csv')
    df_num = preprocessor.preprocess()
    # df_num.reset_index(inplace=True)
    # print(preprocessor.get_unique_users())

    # print(df_num)

    tr = Train(df_num)
    df_tr, df_ts = tr.split()
    print(df_tr)
    # print(df_ts)
    y_tr = df_tr['rating']
    X_tr = df_tr.drop('rating', axis=1)
    y_ts = df_ts['rating']
    X_ts = df_ts.drop('rating', axis = 1)
    print(X_tr)

    print(y_tr)
    # X_tr, X_ts, y_tr, y_ts = tr.split()

if __name__ == '__main__':
    main()