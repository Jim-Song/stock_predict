import torch
from load_data import StocksDataset
from model import Model
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np


data_start_from = 1200
data_end_at = 3000
test_data_start_from = 3000
test_data_end_at = 3499

prices_dir = "datas_clean/prices"
indexes_dir = "datas_clean/indexes"

prices = {}
for file in os.listdir(prices_dir):
    tmp = pd.read_pickle(os.path.join(prices_dir, file))
    if tmp.open.isnull().sum() <= data_start_from and tmp.iloc[-1].isnull().sum() == 0:
        prices[file[: 11]] = tmp


indexes = {}
for file in os.listdir(indexes_dir):
    tmp = pd.read_pickle(os.path.join(indexes_dir, file))
    if tmp.open.isnull().sum() <= data_start_from and tmp.iloc[-1].isnull().sum() == 0:
        indexes[file[: 11]] = tmp



# label_keys = indexes.keys()
# for key in indexes.keys():
#     print(key, indexes[key].open.isnull().sum())
label_keys = ["000133.XSHE"]


stocks_dataset = StocksDataset(prices, 
                               indexes,
                               label_name=label_keys,
                               data_start_from=data_start_from, 
                               data_end_at=data_end_at
                               )

stocks_dataset_test = StocksDataset(prices, 
                                    indexes,
                                    label_name=label_keys,
                                    data_start_from=test_data_start_from, 
                                    data_end_at=test_data_end_at
                                    )

dataloader = DataLoader(stocks_dataset, batch_size=4, shuffle=True, num_workers=4)
dataloader2 = DataLoader(stocks_dataset_test, batch_size=2, shuffle=True, num_workers=4)


model = Model().cuda()
# model.load_state_dict(torch.load("model.pt"))

lr = 8e-2

"""
mean
label
var
probs

"""
step = 0
mean_loss = 0
mean_var = 0
# amp = 1e2
mean_diff_mean = 1e-5

while True:
    test_loss = 0
    sure_degree = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lr *= 0.9999
    for test_batch, item in enumerate(dataloader2):
        # print(step)
        # print(item)
        predict_result = model.forward(item[0], item[1])
        mean = predict_result[:, :7]
        var = predict_result[:, 7:]
        label = item[2].cuda()
        
        # probs = torch.exp(-((mean - label) / var) ** 2) / var
        # loss = torch.mean(-1 * torch.log(probs))
        loss = torch.mean((mean - label) ** 2)
        
        test_loss += loss.detach().item()
        sure_degree += torch.mean(var).item()
    print("test loss: ", step, 
            ", loss: ", test_loss / (test_batch + 1), 
            ", sure_degree: ", sure_degree / (test_batch + 1), 
            ", lr: ", lr,
            )
    
    for _ in range(10):
        for item in dataloader:
            # print(item)
            
            step += 1
            predict_result = model.forward(item[0], item[1])
            # print("0", item[0].isnan().sum())
            # print("1", item[1].isnan().sum())
            mean = predict_result[:, :7]
            var = predict_result[:, 7:]
            label = item[2].cuda()
            # probs = torch.exp(-((mean - label) / var) ** 2) / var
            # loss = torch.mean(-1 * torch.log(probs))
            loss = torch.mean((mean - label) ** 2)
            mean_diff = torch.mean(torch.abs(mean[0] - torch.mean(mean, dim=0)))
            mean_diff_mean = mean_diff_mean * 0.999 + mean_diff.item() * 0.001
            
            if step < 100:
                mean_loss = loss.item()
                mean_var = torch.mean(var).item()
            else: 
                mean_loss = 0.999 * mean_loss + 0.001 * loss.item()
                mean_var = 0.999 * mean_var + 0.001 * torch.mean(var).item()
            if step % 1 == 0:
                print("train loss: ", step, " steps", 
                      ", mean_loss: ", mean_loss, 
                      ", var: ", torch.mean(var).item(), 
                      ", mean_diff_mean", mean_diff_mean,
                      )
                print(mean[0] == mean[1])
                print(mean[: 2])
                print(label[: 2])
            if step % 500 == 0:
                torch.save(model.state_dict(), "model.pt")
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            # for item2 in model.parameters():
            #     print(item2.isnan().sum())
    
            if mean_diff_mean < 1e-5:
                lr *= 1.01
    
        
        # for item2 in model.parameters():
        #     print(item2.isnan().sum())        

    
    
    
    
    