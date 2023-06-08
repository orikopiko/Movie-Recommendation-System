import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df_ratings = pd.read_csv('datasets/ratings.csv')
print(df_ratings)
df_movies = pd.read_csv('datasets/movies.csv')
print(df_movies)
df_tags = pd.read_csv('datasets/tags.csv')
print(df_tags)

df_tags['tag'] = df_tags['tag'].str.lower()
print(df_tags)