'''
#!/usr/bin/python3
 -*- coding: utf-8 -*-
 @Time    : 2019/7/16 22:55
 @Author  : gongkai
 @FileName: main.py
 @Software: PyCharm
 '''
import pandas as pd
import numpy as np
import Get_Feature
import xgboost as xgb

def load_data():
    # 读offline数据
    offline = pd.read_csv(r'data/ccf_offline_stage1_train.csv')
    # 由于存在缺失值，panda会将其转化为floats类型，这里将一些本应该是str的再转为str
    offline.User_id = offline.User_id.astype('str')
    offline.Merchant_id = offline.Merchant_id.astype('str')
    offline.Coupon_id = offline.Coupon_id.astype('str')
    offline.Date_received = offline.Date_received.astype('str')
    # 读test_revised数据
    test = pd.read_csv(r'data/ccf_offline_stage1_test_revised.csv')
    test.User_id = test.User_id.astype('str')
    test.Merchant_id = test.Merchant_id.astype('str')
    test.Coupon_id = test.Coupon_id.astype('str')
    test.Date_received = test.Date_received.astype('str')
    return offline, test


def pro_deal(data):
    data['date_received'] = pd.to_datetime(data['Date_received'], format='%Y%m%d')
    if 'Date' in data.columns.values.tolist():
        data['date'] = pd.to_datetime(data['Date'], format='%Y%m%d')
        data['Date'] = data['Date'].map(lambda x: int(x) if x == x else 0)
    data['isManjian'] = data.Discount_rate.map(
        lambda x: -1 if str(x) == 'null' else (0 if ':' not in str(x) else 1))
    data['Manjian_minCost'] = data.Discount_rate.map(
        lambda x: -1 if (str(x) == 'null' or ':' not in str(x)) else int(str(x).split(':')[0]))
    data['discount_rate'] = data.Discount_rate.map(lambda x: -1 if str(x) == 'null' else (
        float(x) if ':' not in str(x) else (
                (float(str(x).split(':')[0]) - float(str(x).split(':')[1])) / float(str(x).split(':')[0]))))
    data['hasDistance'] = data.Distance.map(lambda x: 1 if x != 'null' else 0)
    data['Distance'].fillna(-1, inplace=True)
    data['distance'] = data.Distance.map(lambda x: int(x))
    data['weekday_Receive'] = data.date_received.map(lambda x: x.weekday())
    # 几号领券
    data['day_Receive'] = data.date_received.map(lambda x: x.day)
    # 工作日领券
    data['isWeekdays_Receive'] = data['weekday_Receive'].map(
        lambda x: 1 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else (0 if (x == 5 or x == 6) else np.nan))
    # 周末领券
    data['isWeekends_Receive'] = data['weekday_Receive'].map(
        lambda x: 1 if (x == 5 or x == 6) else (0 if (x == 0 or x == 1 or x == 2 or x == 3 or x == 4) else np.nan))
    # 缺失值处理
    data['weekday_Receive'].fillna(-1, downcast='infer', inplace=True)  # 填充缺失值
    data['isWeekdays_Receive'].fillna(-1, downcast='infer', inplace=True)  # 填充缺失值
    data['isWeekends_Receive'].fillna(-1, downcast='infer', inplace=True)  # 填充缺失值


def get_label(data):
    data = data[data.Date_received != 'nan']  # 过滤掉领券日期为空的部分
    data.index = range(len(data))  # 重置index
    label = list(map(lambda date, date_received: (date - date_received).days <= 15, data.date,
                     data.date_received))  # 标签为True、False
    label = list(map(int, label))  # True、False标签转为1、0
    data['label'] = label  # 标签列
    data.drop(['Date', 'date'], axis=1, inplace=True)
    return data


def model_xgb(train,test):
    train_label = train['label'].values
    dtrain = xgb.DMatrix(train.drop(['User_id', 'Coupon_id', 'Date_received', 'label'], axis=1).values,
                         label=train_label)
    dtest = xgb.DMatrix(test.drop(['User_id', 'Coupon_id', 'Date_received'], axis=1).values)

    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',
              'silent': 1,
              'gamma': 0,
              'lambda': 1,
              'eta': 0.01,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.8,  # 0.8
              'subsample': 0.8,
              'scale_pos_weight': 1,
              'min_child_weight': 2,  # 2 3
              'nthread': 4
              }
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, num_boost_round=10, evals=watchlist)
    predict = model.predict(dtest)
    test_final = test[['User_id', 'Coupon_id', 'Date_received']]
    test_final['prob'] = predict
    feature_importance = pd.DataFrame(columns=['feature_name', 'importance'])
    feature_importance['feature_name'] = model.get_score().keys()
    feature_importance['importance'] = model.get_score().values()
    feature_importance.sort_values(['importance'], ascending=False, inplace=True)
    return test_final,feature_importance

def model_GBDT(train,test):
    pass
if __name__ == '__main__':
    ## 导数据与预处理
    offline, test_data = load_data()

    pro_deal(offline)
    pro_deal(test_data)
    labeldata = get_label(offline)
    labeldata.User_id = labeldata.User_id.astype('str')
    labeldata.Merchant_id = labeldata.Merchant_id.astype('str')
    labeldata.Coupon_id = labeldata.Coupon_id.astype('str')
    labeldata.Date_received = labeldata.Date_received.astype('str')
    train_get_feature = offline[offline['date_received'].isin(
        pd.date_range('2016/3/2', periods=60))]
    train_null = offline[offline['date_received'].isin(
        pd.date_range('2016/5/2', periods=15))]
    train_have_label = labeldata[labeldata['date_received'].isin(
        pd.date_range('2016/5/16', periods=30))]

    train = train_have_label
    train = Get_Feature.Get_Feature_User_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_Merchant_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_Coupon_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_Discount_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_User_Merchant_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_User_Coupon_Feature(train_have_label, train)
    train = Get_Feature.Get_Feature_User_Discount_Feature(train_have_label, train)

    test_get_feature = offline[offline['date_received'].isin(
        pd.date_range('2016/4/17', periods=60))]
    test_null = offline[offline['date_received'].isin(
        pd.date_range('2016/6/18', periods=15))]
    test_real = test_data

    test = test_real
    test = Get_Feature.Get_Feature_User_Feature(test_real, test)
    test = Get_Feature.Get_Feature_Merchant_Feature(test_real, test)
    test = Get_Feature.Get_Feature_Coupon_Feature(test_real, test)
    test = Get_Feature.Get_Feature_Discount_Feature(test_real, test)
    test = Get_Feature.Get_Feature_User_Merchant_Feature(test_real, test)
    test = Get_Feature.Get_Feature_User_Coupon_Feature(test_real, test)
    test = Get_Feature.Get_Feature_User_Discount_Feature(test_real, test)
    test.drop(['Merchant_id', 'Discount_rate', 'Distance', 'date_received'], axis=1, inplace=True)
    train.drop(['Merchant_id', 'Discount_rate', 'Distance', 'date_received'], axis=1, inplace=True)
    print(train.shape)
    print(test.shape)
    predict,feature_important = model_xgb(train,test)
    predict.rename(columns={0: 'prob'}, inplace=True)
    predict.to_csv(r'result4.csv', index=False)