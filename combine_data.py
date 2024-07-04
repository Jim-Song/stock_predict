import pandas as pd
import os
import datetime
import numpy as np




def combine(datadir_list, output_dir):
    
    files_list = []
    for datadir in datadir_list:
        files_list += os.listdir(datadir)
    files_set = set(files_list)

    for file in files_set:
        print(file)
        data_tmp_list = []
        for datadir in datadir_list:
            if file in os.listdir(datadir):
                data_tmp_list.append(pd.read_pickle(os.path.join(datadir, file)))
        if len(data_tmp_list) < len(datadir_list):
            continue
        combined_df = pd.concat(data_tmp_list, axis=0)
        combined_df.volume.replace(0, np.nan, inplace=True)
        if np.isnan(combined_df.iloc[-20].money).sum() > 0:
            print("最近20个交易日有停牌，不选择")
            continue
        combined_df.volume.fillna(method='bfill', inplace=True)
        combined_df.money.replace(0, np.nan, inplace=True)
        combined_df.money.fillna(method='bfill', inplace=True)
        # combined_df['volume'].fillna(0, inplace=True)
        # combined_df['money'].fillna(0, inplace=True)
        combined_df['volume'] = combined_df['volume'].rolling(window=10, min_periods=1).mean()
        combined_df['money'] = combined_df['money'].rolling(window=10, min_periods=1).mean()
        combined_df.to_pickle(os.path.join(output_dir, file))
        


datadir_list = ["datas/orig/prices", "datas/0701/prices", "datas/0705/prices", ]
output_dir = "datas_clean2/prices"
combine(datadir_list, output_dir)
datadir_list = ["datas/orig/indexes", "datas/0701/indexes", "datas/0705/indexes", ]
output_dir = "datas_clean2/indexes"
combine(datadir_list, output_dir)

