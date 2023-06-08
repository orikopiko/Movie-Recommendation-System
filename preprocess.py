import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df_ratings = pd.read_csv('datasets/ratings.csv')
# print(df_ratings)
df_movies = pd.read_csv('datasets/movies.csv')
# print(df_movies)
df_tags = pd.read_csv('datasets/tags.csv')
# print(df_tags)
df_tags['tag'] = df_tags['tag'].str.lower()
# print(df_tags)

print(df_movies)
genre_names = ["Action", "Adventure", "Animation", "Children\'s","Comedy","Crime","Documentary","Drama","Fantasy",
               "Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western",
               "(no genres listed)"]

genres = df_movies['genres'].str.get_dummies('|')
# Concatenate the encoded genres back to the original DataFrame
df_movies = pd.concat([df_movies, genres], axis=1)

# Drop the original genre column
df_movies.drop('genres', axis=1, inplace=True)
