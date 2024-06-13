import pandas as pd
import os
import datetime
import numpy as np


for file in os.listdir("datas_clean/prices"):
    price = pd.read_pickle(os.path.join("datas_clean/prices", file))
    for i in range(len(price)):
        if price.iloc[i].volume == 0:
            for j in range(1000):
                if not price.iloc[i - j].volume == 0:
                    price.iloc[i].volume = price.iloc[i - j].volume
                    price.iloc[i].money = price.iloc[i - j].money
                    break
    
    price.to_pickle("datas_clean/prices/" + file)
    


for file in os.listdir("datas_clean/indexes"):
    index = pd.read_pickle(os.path.join("datas_clean/indexes", file))
    for i in range(len(index)):
        if index.iloc[i].volume == 0:
            for j in range(1000):
                if not index.iloc[i - j].volume == 0:
                    index.iloc[i].volume = index.iloc[i - j].volume
                    index.iloc[i].money = index.iloc[i - j].money
                    break
                
    index.to_pickle("datas_clean/indexes/" + file)






