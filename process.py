import lightgbm as lgb
import pandas as pd

def training(X_train, Y_train, X_valid, Y_valid):
    feature_name = X_train.columns.tolist()
    params = {
        'objective': 'mse',
        'metric': 'rmse',
        'num_leaves': 2 ** 7 - 1,
        'learning_rate': 0.005,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'seed': 1,
        'verbose': 1
    }


    lgb_train = lgb.Dataset(X_train[feature_name], Y_train)
    lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)

    evals_result = {}
    gbm = lgb.train(
            params, 
            lgb_train,
            num_boost_round=3000,
            valid_sets=(lgb_train, lgb_eval), 
            feature_name = feature_name,
            verbose_eval=5, 
            evals_result = evals_result,
            early_stopping_rounds = 100)
    return gbm

def testing(gbm, X_train, X_test):
    feature_name = X_train.columns.tolist()
    test = pd.read_csv('./data/test.csv')
    Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)

    submission = pd.DataFrame({
        "ID": test.index, 
        "item_cnt_month": Y_test
    })
    submission.to_csv('gbm_submission.csv', index=False)