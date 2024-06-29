import pandas as pd
import os
import datetime
import numpy as np


for file in os.listdir("datas_clean/prices"):
    price = pd.read_pickle(os.path.join("datas_clean/prices", file))
    for i in range(len(price)):
        # current_index = len(price)
        # 部分volumn缺失，使用上一天的volumn
        if price.iloc[i].volume == 0:
            for j in range(1000):
                if not price.iloc[i - j].volume == 0:
                    price.iloc[i].volume = price.iloc[i - j].volume
                    price.iloc[i].money = price.iloc[i - j].money
                    break
        # # volume 值方差过大，使用历史平均值平滑
        # range_len = min(10, i)
        # price.iloc[i].volume = price.iloc[i - range_len: i + 1].volume.mean()
        # price.iloc[i].money = price.iloc[i - range_len: i + 1].money.mean()
    
    price.to_pickle("datas_clean/prices/" + file)
    print(file)
    


for file in os.listdir("datas_clean/indexes"):
    index = pd.read_pickle(os.path.join("datas_clean/indexes", file))
    for i in range(len(index)):
        if index.iloc[i].volume == 0:
            for j in range(1000):
                if not index.iloc[i - j].volume == 0:
                    index.iloc[i].volume = index.iloc[i - j].volume
                    index.iloc[i].money = index.iloc[i - j].money
                    break
        # range_len = min(10, i)
        # index.iloc[i].volume = index.iloc[i - range_len: i + 1].volume.mean()
        # index.iloc[i].money = index.iloc[i - range_len: i + 1].money.mean()
                
    index.to_pickle("datas_clean/indexes/" + file)
    print(file)
    

file = "399986.XSHE.pkl"
index = pd.read_pickle(os.path.join("datas_clean/indexes", file))



