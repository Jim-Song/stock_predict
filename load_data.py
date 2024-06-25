# 数据：
# open close high low quantity volumn


import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

import numpy as np
import json
import os
import copy


    
# 自定义数据集类
class StocksDataset(Dataset):
    def __init__(self, 
                 prices, 
                 indexes, 
                 label_name=["000133.XSHG"], 
                 data_start_from=1158, 
                 data_end_at=3000, 
                 num1=500, 
                 num2=100,
                 ):
        time_slices = [0, -1,  -2,  -3,  -4,  -5,  -7,  -9,  -11, -13, -15, 
                       -18, -21, -24, -27, -30, -34, -38, -42, -46, -50, 
                       -55, -60, -65, -70, -75, -81, -87, -93, -99, -105, -112, -119, -126, -133, -143, -155]
        time_slices_label = [0, 1, 2, 4, 8, 16, 28, 44, 64]
        self.history_len = len(time_slices)
        self.predict_len = len(time_slices_label)
        self.data_start_from = data_start_from
        self.data_end_at = data_end_at
        self.time_range = time_slices_label[-1] - time_slices[-1]
        self.label_name = label_name
        self.label_data = None
        self.label_mean = None
        self.label_std = None
        self.num1 = num1
        self.num2 = num2
        self.key_indexes_dict = {}
        self.key_indexes_list = []
        
        self.time_slices = np.array(time_slices)
        self.time_slices_label = np.array(time_slices_label)

        prices_and_indexes = copy.copy(prices)
        prices_and_indexes.update(indexes)
        self.prices_and_indexes_data_block = self.get_data_block(prices_and_indexes)
        # mean = np.nanmean(self.prices_data_block, axis=1)
        # mean = np.expand_dims(mean, 1)
        # std = np.nanstd(self.prices_data_block, axis=1)
        # std = np.expand_dims(std, 1)
        # self.prices_data_block = (self.prices_data_block - mean) / std
        
        # self.indexes_data_block = self.get_data_block(indexes)
        # mean = np.nanmean(self.indexes_data_block, axis=1)
        # mean = np.expand_dims(mean, 1)
        # std = np.nanstd(self.indexes_data_block, axis=1)
        # std = np.expand_dims(std, 1)
        # self.indexes_data_block = (self.indexes_data_block - mean) / std
        
        could_train_len = self.prices_and_indexes_data_block.shape[1] - self.time_range - 1
        print("could_train_len: ", could_train_len)
        self.data = list(range(could_train_len))
        if self.num1 == 0:
            self.num1 = len(self.prices_and_indexes_data_block)
        # if self.num2 == 0:
        #     self.num2 = len(self.indexes_data_block)
    
    def get_data_block(self, prices):
        filtered_st_prices = {}
        for key, value in prices.items():
            if value.iloc[-1].isnull().sum() > 0:
                continue
            value = value[self.data_start_from: self.data_end_at]
            filtered_st_prices[key] = value
        filtered_st_prices_list = []
        means = {}
        stds = {}
        ct = 0
        for key1, value in filtered_st_prices.items():
            print(ct, key1)
            filtered_st_prices_list.append(np.expand_dims(value.values, axis=0))
            if key1 in self.label_name:
                self.key_indexes[key1] = ct
            ct += 1
            #     self.label_data = filtered_st_prices_list[-1]
                # self.label_mean = tmp_mean
                # self.label_std = tmp_std
        # print("key indexes: ", self.key_indexes)
        prices_with_nan = np.concatenate(filtered_st_prices_list, axis=0)
        prices_with_nan = prices_with_nan[:, :, :]
        # prices_with_nan = torch.tensor(prices_with_nan)
        # prices_with_nan = torch.transpose(prices_with_nan, 0, 1)
        return prices_with_nan
    
    
    def get_non_nan_data_row(self, data_block, idx, bias, num, must_have_indexes=[]):
        # print(idx)
        # num = 100
        # bias = 0
        # must_have_indexes=[]
        # data_block = prices_with_nan[:, 1200:, :]
        stocks_num = data_block.shape[0]
        idx = idx - self.time_slices[-1]
        # tmp = data_block[:, self.time_slices + idx, :]
        slice_all = []
        for i in range(self.history_len - 1):
            # start = self.time_slices[i]
            # end = self.time_slices[i + 1]
            slice_tmp = np.mean(data_block[:, self.time_slices[i + 1] + idx: self.time_slices[i] + idx, :], axis=1)
            slice_tmp = np.expand_dims(slice_tmp, axis=1)
            slice_all.append(slice_tmp)
        tmp = np.concatenate(slice_all, axis=1)
        tmp2 = copy.copy(tmp)
        
        rand_vars = np.linspace(0.01, 0.04, self.history_len - 1)
        stocks_num, time_range, feature_size = tmp.shape
        rands = []
        for var in rand_vars:
            rands.append(np.random.normal(1, var, (stocks_num, 1, feature_size)))
        rand_array = np.concatenate(rands, axis=1)
        tmp = tmp * rand_array

# tmp = raw_prices[0].numpy()
        jizhun = tmp[:, 0, :].copy()
        jizhun = np.expand_dims(jizhun, axis=1)
        tmp = (tmp / jizhun) - 1
        tmp = tmp[:, 1:, :]
        # stock, time, price
        tmp_mean = np.mean(tmp, axis=0)
        tmp_mean = np.expand_dims(tmp_mean, axis=0)
        tmp_std = np.std(tmp, axis=0)
        tmp_std = np.expand_dims(tmp_std, axis=0)
        tmp = (tmp - tmp_mean) / tmp_std

        stocks_indexes = np.array(list(range(stocks_num)))
        stock_not_nan_indexes = ~np.isnan(tmp).any(axis=2).any(axis=1)
        stock_not_nan_indexes = stocks_indexes[stock_not_nan_indexes]
        # print(stock_not_nan_indexes)
        # print(idx)
        # print(num)
        stock_final_indexes = np.random.choice(stock_not_nan_indexes, num, replace=False)
        if must_have_indexes:
            stock_final_indexes = [item for item in stock_final_indexes if item not in must_have_indexes]
            stock_final_indexes = np.concatenate([must_have_indexes, stock_final_indexes])[: num]
        # print(stock_final_indexes)
        stock_final_indexes.sort()
        stocks_final = tmp[stock_final_indexes, :, :]
        # stocks_final = torch.tensor(stocks_final, dtype=torch.float32).cuda()
        stock_final_indexes += bias

        return stocks_final, stock_final_indexes, tmp2


    def get_label(self, data_block, idx, must_have_indexes=[0]):
        idx = idx - self.time_slices[-1]
        slice_all = []
        for i in range(self.predict_len - 1):
            # start = self.time_slices_label[i]
            # end = self.time_slices_label[i + 1]
            slice_tmp = np.mean(data_block[:, self.time_slices_label[i] + idx: self.time_slices_label[i + 1] + idx, :], axis=1)
            slice_tmp = np.expand_dims(slice_tmp, axis=1)
            slice_all.append(slice_tmp)
            
        tmp = np.concatenate(slice_all, axis=1)
        tmp = tmp[must_have_indexes]
        jizhun = tmp[:, 0, :].copy()
        jizhun = np.expand_dims(jizhun, axis=1)
        tmp = (tmp / jizhun) - 1
        tmp = tmp[:, 1:, :]
        # stock, time, price
        tmp_mean = np.mean(tmp, axis=0)
        tmp_mean = np.expand_dims(tmp_mean, axis=0)
        tmp_std = np.std(tmp, axis=0)
        tmp_std = np.expand_dims(tmp_std, axis=0)
        tmp = (tmp - tmp_mean) / tmp_std

        stocks_final = tmp[:, :, 0]
        return stocks_final



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        stock_prices, stock_indexes, stock_prices_raw = self.get_non_nan_data_row(self.prices_and_indexes_data_block, idx, bias=0, num=self.num1)
        # index_prices, index_indexes = self.get_non_nan_data_row(self.indexes_data_block, idx, bias=10000, num=self.num2, must_have_indexes=[])
        
        label_prices = self.get_label(self.prices_and_indexes_data_block, idx, must_have_indexes=stock_indexes)
        
        # normal
        sample1 = stock_prices
        # sample1 = np.concatenate([stock_prices[:, :, :], index_prices[:, :, :]], axis=0)
        # rand = np.random.normal(1, 0.02, size=sample1.shape)
        # sample1 = sample1 * rand
        # cheat
        # sample1 = np.concatenate([stock_prices, index_prices], axis=0)

        # sample1 = index_prices[0: 2, :, :]
        # sample1 = np.expand_dims(sample1, axis=0)
        # jizhun = sample1[:, 0, :].copy()
        # jizhun = np.expand_dims(jizhun, axis=1)
        # jizhun[:, :, :4] = 1
        # print(jizhun)
        # sample1 = (sample1 / jizhun) - 1
        # sample2 = np.concatenate([stock_indexes, index_indexes], axis=0)
        sample2 = np.array(stock_indexes)
        # sample2 = np.array([0, 0])
        # label = index_prices[0, self.history_len:, 0] / index_prices[0, 0, 0]
        sample1 = torch.tensor(sample1, dtype=torch.float32)
        sample1 = sample1[:, :, :4]
        return sample1, sample2, label_prices, idx, stock_prices_raw









