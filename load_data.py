# 数据：
# open close high low quantity volumn


import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

import numpy as np
import json
import os


    
# 自定义数据集类
class StocksDataset(Dataset):
    def __init__(self, 
                 prices, 
                 indexes, 
                 label_name="000133.XSHG", 
                 data_start_from=1158, 
                 data_end_at=3000, 
                 ):
        time_slices = [0, -1,  -2,  -3,  -4,  -5,  -7,  -9,  -11, -13, -15, 
                       -18, -21, -24, -27, -30, -34, -38, -42, -46, -50, 
                       -55, -60, -65, -70, -75, -81, -87, -93, -99, -105, -112, -119, -126, -133, 
                       1, 2, 4, 8, 16, 32, 64]
        self.history_len = 35
        self.predict_len = 7
        self.data_start_from = data_start_from
        self.data_end_at = data_end_at
        self.time_range = 64 + 133
        self.label_name = label_name
        self.label_data = None
        self.label_mean = None
        self.label_std = None
        
        self.time_slices = np.array(time_slices)


        self.prices_data_block = self.get_data_block(prices)
        # mean = np.nanmean(self.prices_data_block, axis=1)
        # mean = np.expand_dims(mean, 1)
        # std = np.nanstd(self.prices_data_block, axis=1)
        # std = np.expand_dims(std, 1)
        # self.prices_data_block = (self.prices_data_block - mean) / std
        
        self.indexes_data_block = self.get_data_block(indexes)
        # mean = np.nanmean(self.indexes_data_block, axis=1)
        # mean = np.expand_dims(mean, 1)
        # std = np.nanstd(self.indexes_data_block, axis=1)
        # std = np.expand_dims(std, 1)
        # self.indexes_data_block = (self.indexes_data_block - mean) / std
        
        could_train_len = self.prices_data_block.shape[1] - self.time_range - 1
        print("could_train_len: ", could_train_len)
        self.data = list(range(could_train_len))
    
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
        for key1, value in filtered_st_prices.items():
            print(key1)
            filtered_st_prices_list.append(np.expand_dims(value.values, axis=0))
            if key1 in self.label_name:
                self.label_data = filtered_st_prices_list[-1]
                self.label_mean = tmp_mean
                self.label_std = tmp_std
                
        prices_with_nan = np.concatenate(filtered_st_prices_list, axis=0)
        prices_with_nan = prices_with_nan[:, :, :]
        # prices_with_nan = torch.tensor(prices_with_nan)
        # prices_with_nan = torch.transpose(prices_with_nan, 0, 1)
        return prices_with_nan
    
    def get_non_nan_data_row(self, data_block, idx, bias, num):
        stocks_num = data_block.shape[0]
        idx = idx + 133
        tmp = data_block[:, self.time_slices + idx, :]
        stocks_indexes = np.array(list(range(stocks_num)))
        stock_not_nan_indexes = ~np.isnan(tmp).any(axis=2).any(axis=1)
        stock_not_nan_indexes = stocks_indexes[stock_not_nan_indexes]
        stock_final_indexes = np.random.choice(stock_not_nan_indexes, num)
        stocks_final = tmp[stock_final_indexes, :, :]
        # stocks_final = torch.tensor(stocks_final, dtype=torch.float32).cuda()
        stock_final_indexes += bias
        return stocks_final, stock_final_indexes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        stock_prices, stock_indexes = self.get_non_nan_data_row(self.prices_data_block, idx, bias=0, num=811)
        index_prices, index_indexes = self.get_non_nan_data_row(self.indexes_data_block, idx, bias=10000, num=200)
        
        sample1 = np.concatenate([stock_prices[:, :self.history_len, :], index_prices[:, :self.history_len, :]], axis=0)
        jizhun = sample1[:, 0, :]
        jizhun = np.expand_dims(jizhun, axis=1)
        jizhun[:, :, :4] = 1
        # print(jizhun)
        sample1 = sample1 / jizhun
        sample2 = np.concatenate([stock_indexes, index_indexes], axis=0)
        label = index_prices[0, self.history_len:, 0] / 10000 # / index_prices[0, 0, 0]
        sample1 = torch.tensor(sample1, dtype=torch.float32)
        return sample1, sample2, label









