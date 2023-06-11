from preprocess import Preprocessor
def main():
    preprocessor = Preprocessor('datasets/ratings.csv', 'datasets/movies.csv', 'datasets/tags.csv')
    df_num = preprocessor.preprocess()
    print(df_num)

if __name__ == '__main__':
    main()