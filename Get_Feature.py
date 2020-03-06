'''
#!/usr/bin/python3
 -*- coding: utf-8 -*-
 @Time    : 2019/7/18 9:10
 @Author  : gongkai
 @FileName: Get_Feature.py
 @Software: PyCharm
 '''
import pandas as pd
import numpy as np


def Get_Feature_User_Feature(data, dataset):
    # 主键
    keys = list(['User_id'])
    # 特征名前缀
    prefixs = '_'.join(keys)+'_'
    # 领券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=len)
    #inplace = True：不创建新的对象，直接对原始对象进行修改；
    #inplace = False：对数据进行修改，创建并返回新的对象承载其修改结果。
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 领不同类券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_different_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_different_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 领不同商家次数
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Merchant_id': prefixs + 'Receive_different_Merchant_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_different_Merchant_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='Coupon_id',
                           aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    # 领取满减券次数
    pivot = pd.pivot_table(data[data.isManjian == 1], index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_isManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_isManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取满减券
    # 领取非满减券次数
    pivot = pd.pivot_table(data[data.isManjian != 1], index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_isNotManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_isNotManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取非满减券
    # 领取满减券次数 / 领券次数
    dataset[prefixs + 'Receive_Coupon_cnt_isManjian_div_Receive_cnt'] = dataset[
                                                                            prefixs + 'Receive_Coupon_cnt_isManjian'] / (
                                                                                    dataset[
                                                                                        prefixs + 'Receive_Coupon_cnt'] + 0.1)
    # 领取非满减次数 / 领券次数
    dataset[prefixs + 'Receive_Coupon_cnt_isNotManjian_div_Receive_cnt'] = dataset[
                                                                               prefixs + 'Receive_Coupon_cnt_isNotManjian'] / (
                                                                                       dataset[
                                                                                           prefixs + 'Receive_Coupon_cnt'] + 0.1)
    return dataset


def Get_Feature_Merchant_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['Merchant_id'])
    # 特征名前缀
    prefixs='_'.join(keys)+'_'
    # 商家被领取的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 商家被不同用户领取的优惠券数目
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'User_id': prefixs + 'Received_different_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_different_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 商家优惠券种类
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Coupon_id': prefixs + 'different_Conpon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'different_Conpon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬被领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='User_id',
                           aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    # 被领取满减券次数
    pivot = pd.pivot_table(data[data.isManjian == 1], index=keys, values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_isManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_isManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取满减券
    # 被领取非满减券次数
    pivot = pd.pivot_table(data[data.isManjian != 1], index=keys, values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_isNotManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_isNotManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取非满减券
    # 领取满减券次数 / 领券次数
    dataset[prefixs + 'Received_User_cnt_isManjian_div_Receive_cnt'] = dataset[
                                                                           prefixs + 'Received_User_cnt_isManjian'] / (
                                                                                   dataset[
                                                                                       prefixs + 'Received_User_cnt'] + 0.1)
    # 领取非满减次数 / 领券次数
    dataset[prefixs + 'Received_User_cnt_isNotManjian_div_Receive_cnt'] = dataset[
                                                                              prefixs + 'Received_User_cnt_isNotManjian'] / (
                                                                                      dataset[
                                                                                          prefixs + 'Received_User_cnt'] + 0.1)

    return dataset


def Get_Feature_Coupon_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['Coupon_id'])
    # 特征名前缀
    prefixs = '_'.join(keys) + '_'
    # 被领取的数目
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 被多少不同用户领取
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'User_id': prefixs + 'Received_different_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_different_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬被领次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='User_id',
                           aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券

    return dataset


def Get_Feature_Discount_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['Discount_rate'])
    # 特征名前缀
    prefixs = '_'.join(keys) + '_'
    # 被多少用户领
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 被多少不同用户领
    pivot = pd.pivot_table(data, index=keys, values='User_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'User_id': prefixs + 'Received_different_User_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_different_User_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 被多少商家发放
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Merchant_id': prefixs + 'different_Merchant_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'different_Merchant_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬被领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='User_id', aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='User_id',
                           aggfunc=len)
    pivot.rename(columns={'User_id': prefixs + 'Received_User_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Received_User_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    return dataset


def Get_Feature_User_Merchant_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['User_id', 'Merchant_id'])
    # 特征名前缀
    prefixs = '_'.join(keys) + '_'
    # 领券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 领不同类券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_different_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_different_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='Coupon_id',
                           aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    # 领取满减券次数
    pivot = pd.pivot_table(data[data.isManjian == 1], index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_isManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_isManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取满减券
    # 领取非满减券次数
    pivot = pd.pivot_table(data[data.isManjian != 1], index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_isNotManjian'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_isNotManjian'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领取非满减券
    # 领取满减券次数 / 领券次数
    dataset[prefixs + 'Receive_Coupon_cnt_isManjian_div_Receive_cnt'] = dataset[
                                                                            prefixs + 'Receive_Coupon_cnt_isManjian'] / (
                                                                                    dataset[
                                                                                        prefixs + 'Receive_Coupon_cnt'] + 0.1)
    # 领取非满减次数 / 领券次数
    dataset[prefixs + 'Receive_Coupon_cnt_isNotManjian_div_Receive_cnt'] = dataset[
                                                                               prefixs + 'Receive_Coupon_cnt_isNotManjian'] / (
                                                                                       dataset[
                                                                                           prefixs + 'Receive_Coupon_cnt'] + 0.1)

    return dataset


def Get_Feature_User_Coupon_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['User_id', 'Coupon_id'])
    # 特征名前缀
    prefixs = '_'.join(keys) + '_'
    # 领券次数
    pivot = pd.pivot_table(data, index=keys, values='Discount_rate', aggfunc=len)
    pivot.rename(columns={'Discount_rate': prefixs + 'Receive_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='Discount_rate', aggfunc=len)
    pivot.rename(columns={'Discount_rate': prefixs + 'Receive_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='Discount_rate', aggfunc=len)
    pivot.rename(columns={'Discount_rate': prefixs + 'Receive_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='Discount_rate',
                           aggfunc=len)
    pivot.rename(columns={'Discount_rate': prefixs + 'Receive_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    return dataset


def Get_Feature_User_Discount_Feature(dataframe_L, dataframe):
    data = dataframe_L.copy()  # 提特征的集
    dataset = dataframe.copy()  # 已构造好的集
    # 主键
    keys = list(['User_id', 'Discount_rate'])
    # 特征名前缀
    prefixs = '_'.join(keys) + '_'
    # 领券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 领不同类券次数
    pivot = pd.pivot_table(data, index=keys, values='Coupon_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_different_Coupon_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_different_Coupon_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 领不同商家次数
    pivot = pd.pivot_table(data, index=keys, values='Merchant_id', aggfunc=lambda x: len(set(x)))
    pivot.rename(columns={'Merchant_id': prefixs + 'Receive_different_Merchant_cnt'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_different_Merchant_cnt'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示没领券
    # 上旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 1 and x.day < 11)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_early'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_early'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示上旬没领券
    # 中旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 11 and x.day < 21)], index=keys,
                           values='Coupon_id', aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_middle'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_middle'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示中旬没领券
    # 下旬领券次数
    pivot = pd.pivot_table(data[data.date_received.map(lambda x: x.day >= 21)], index=keys, values='Coupon_id',
                           aggfunc=len)
    pivot.rename(columns={'Coupon_id': prefixs + 'Receive_Coupon_cnt_late'}, inplace=True)
    dataset = pd.merge(dataset, pivot, left_on=keys, right_index=True, how='left')
    dataset[prefixs + 'Receive_Coupon_cnt_late'].fillna(0, downcast='infer', inplace=True)  # 填缺失值0表示下旬没领券
    return dataset
