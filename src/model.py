import pandas as pd
import numpy as np


df_train = pd.read_csv('data/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('data/test.csv', dtype={'id': np.int16})
if __name__ == "__main__":
    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))