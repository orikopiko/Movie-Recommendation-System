import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
class Preprocessor:
    def __init__(self, ratingFile, movieFile, tagsFile):
        self.ratingFile = ratingFile
        self.movieFile = movieFile
        self.tagsFile = tagsFile
        self.df_ratings = None
        self.df_movies = None
        self.df_tags = None
        self.df_numeric = None
        self.df_all = None

    def load_data(self):
        self.df_ratings = pd.read_csv(self.ratingFile)
        self.df_movies = pd.read_csv(self.movieFile)
        self.df_tags = pd.read_csv(self.tagsFile)
    def clean_data(self):
        self.df_tags['tag'] = self.df_tags['tag'].str.lower()
        genre_names = ["Action", "Adventure", "Animation", "Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
               "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",
               "(no genres listed)"]
        genres = self.df_movies['genres'].str.get_dummies('|')
        # Concatenate the encoded genres back to the original DataFrame
        self.df_movies = pd.concat([self.df_movies, genres], axis=1)

        # Drop the original genre column
        self.df_movies.drop('genres', axis=1, inplace=True)
        self.df_ratings.drop('timestamp', axis=1, inplace=True)
        self.df_tags.drop('timestamp', axis=1, inplace=True)
        columns_to_add = self.df_movies.columns[self.df_movies.columns != 'title']
        # print(columns_to_add)
        self.df_numeric = self.df_ratings.merge(self.df_movies[columns_to_add], on='movieId')
        self.df_numeric = self.df_numeric.sort_values(by='userId')
    def get_unique_users(self):
        self.load_data()
        self.clean_data()
        max_user_id = self.df_numeric['userId'].iloc[-1]
        num_entries_per_user = np.zeros((max_user_id,))
        for i in range(1, max_user_id+1):
            num_entries_per_user[i-1] = np.count_nonzero(self.df_numeric['userId'] == i)
        return num_entries_per_user
    def preprocess(self):
        self.load_data()
        self.clean_data()
        
        return self.df_numeric

# df_ratings = pd.read_csv('datasets/ratings.csv')
# # print(df_ratings.info())
# df_movies = pd.read_csv('datasets/movies.csv')
# # print(df_movies)
# df_tags = pd.read_csv('datasets/tags.csv')
# # print(df_tags)
# df_tags['tag'] = df_tags['tag'].str.lower()
# print(df_tags.info())

# # print(df_movies)
# genre_names = ["Action", "Adventure", "Animation", "Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
#                "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",
#                "(no genres listed)"]

# genres = df_movies['genres'].str.get_dummies('|')
# # Concatenate the encoded genres back to the original DataFrame
# df_movies = pd.concat([df_movies, genres], axis=1)

# # Drop the original genre column
# df_movies.drop('genres', axis=1, inplace=True)
# df_ratings.drop('timestamp', axis=1, inplace=True)
# df_tags.drop('timestamp', axis=1, inplace=True)

# # print('movies:')
# # print(df_movies)

# columns_to_add = df_movies.columns[df_movies.columns != 'title']
# # print(columns_to_add)
# new_df = df_ratings.merge(df_movies[columns_to_add], on='movieId')
# new_df = new_df.sort_values(by='userId')
# new_df.columns = new_df.columns.str.strip()
# df_tags.columns = df_tags.columns.str.strip()

# df_all = new_df.merge(df_tags[df_tags.columns], on=['movieId', 'userId'], how='left')
# print(df_all[df_all['tag'].notnull()])
# # print('ratings')
# # print(df_ratings.sort_values(by='movieId'))
# # print(df_tags)
# # print(new_df)
# # print(new_df[new_df['movieId'] == 193609])
# # print(df_movies[df_movies['(no genres listed)'] == 1])

# max_user_id = new_df['userId'].iloc[-1]
# num_entries_per_user = np.zeros((max_user_id,))
# for i in range(1, max_user_id+1):
#     num_entries_per_user[i-1] = np.count_nonzero(new_df['userId'] == i)
# # print(num_entries_per_user)
# # print(np.sum(num_entries_per_user)) # matches overall dataframe row number. Means we are not missing any user