
import gc
import re
import string
import operator
from collections import defaultdict

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import matplotlib.pyplot as plt
import seaborn as sns

import tokenization
from wordcloud import STOPWORDS

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
# import model

df_train = pd.read_csv('/content/drive/MyDrive/Kaggle/disaster_tweets/train.csv', dtype={'id': np.int16, 'target': np.int8})
df_test = pd.read_csv('/content/drive/MyDrive/Kaggle/disaster_tweets/test.csv', dtype={'id': np.int16})
if __name__ == "__main__":
    print('Training Set Shape = {}'.format(df_train.shape))
    print('Training Set Memory Usage = {:.2f} MB'.format(df_train.memory_usage().sum() / 1024**2))
    print('Test Set Shape = {}'.format(df_test.shape))
    print('Test Set Memory Usage = {:.2f} MB'.format(df_test.memory_usage().sum() / 1024**2))
# import tokenization
# from wordcloud import STOPWORDS

# from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
# from sklearn.metrics import precision_score, recall_score, f1_score

# import tensorflow as tf
# import tensorflow_hub as hub
# from tensorflow import keras
# from tensorflow.keras.optimizers import SGD, Adam
# from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling1D
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback


missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x= df_train[missing_cols].isnull().sum().index, y= df_train[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x= df_test[missing_cols].isnull().sum().index, y= df_test[missing_cols].isnull().sum().values, ax=axes[1])
axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)
print(f"Percentage of missing data in Train: {100 *  df_train[missing_cols].isna().sum() /  df_train[missing_cols].count()}" )
print(f"Percentage of missing data in Test: {100 *  df_test[missing_cols].isna().sum() /  df_test[missing_cols].count()}" )
# plt.show()
print(  df_train[missing_cols].count())
print( df_train[missing_cols].isna().sum())

for df in [ df_train,  df_test]:
    for col in ['keyword', 'location']:
        df[col] = df[col].fillna(f'no_{col}')


print(f'Number of unique values in keyword = { df_train["keyword"].nunique()} (Training) - { df_test["keyword"].nunique()} (Test)')
print(f'Number of unique values in location = { df_train["location"].nunique()} (Training) - { df_test["location"].nunique()} (Test)')


df_train['target_mean'] =  df_train.groupby('keyword')['target'].transform('mean')

fig = plt.figure(figsize=(8, 72), dpi=100)

sns.countplot(y= df_train.sort_values(by='target_mean', ascending=False)['keyword'],
              hue= df_train.sort_values(by='target_mean', ascending=False)['target'])

plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')

plt.show()

df_train.drop(columns=['target_mean'], inplace=True)