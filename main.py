
from data_preprocess import Features
from process import *
import pandas as pd

def readFeature():
    df = pd.read_pickle('df.pkl')
    df.info()
    X_train = df[df.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = df[df.date_block_num < 33]['item_cnt_month']
    X_valid = df[df.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = df[df.date_block_num == 33]['item_cnt_month']
    X_test = df[df.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    del df
    return X_train, Y_train, X_valid, Y_valid, X_test


if __name__ =='__main__':
    f = Features()
    f.execute()
    X_train, Y_train, X_valid, Y_valid, X_test = readFeature()
    gbm = training(X_train, Y_train, X_valid, Y_valid)
    testing(gbm, X_train, X_test)