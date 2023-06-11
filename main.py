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
    X_tr, X_ts, y_tr, y_ts = tr.split()
    tr.fit_models()

if __name__ == '__main__':
    main()