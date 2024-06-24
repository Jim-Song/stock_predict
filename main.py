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
                               data_end_at=data_end_at, 
                               num1=2, 
                               num2=2,
                               )

stocks_dataset_test = StocksDataset(prices, 
                                    indexes,
                                    label_name=label_keys,
                                    data_start_from=test_data_start_from, 
                                    data_end_at=test_data_end_at,
                                    num1=800, 
                                    num2=200,
                                    )

dataloader = DataLoader(stocks_dataset, batch_size=32, shuffle=True, num_workers=32)
dataloader2 = DataLoader(stocks_dataset_test, batch_size=2, shuffle=True, num_workers=5)


model = Model().cuda()
model.load_state_dict(torch.load("model.pt"))

lr = 2e-5

"""
mean
label
(mean - label) ** 2
var
probs

"""
step = 0
mean_loss = 0
mean_var = 0
amp = 1
mean_diff_mean = 4e-4

train_loss_comp = 0.0021
train_data_mean = torch.tensor([1.0] * 7).cuda()

while True:
    test_loss = 0
    test_loss_comp = 0
    sure_degree = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-10)
    lr *= 0.98
    
    model.eval()
    test_batch_restore = None
    for test_batch, item in enumerate(dataloader2):
        if not test_batch_restore:
            test_batch_restore = item
        # print(step)
        # print(item)
        if item[0].shape[0] <= 1:
            continue
        predict_result = model.forward(item[0], item[1])
        mean = predict_result[:, :7]
        var = predict_result[:, 7:]
        label = item[2].cuda()
        
        # probs = torch.exp(-((mean - label) / var) ** 2) / var
        # loss = torch.mean(-1 * torch.log(probs))
        
        loss = torch.mean((mean - label) ** 2)
        ones = torch.ones(mean.shape).cuda()
        test_loss_comp += torch.mean((ones - label) ** 2).detach().item()
        
        test_loss += loss.detach().item()
        sure_degree += torch.mean(var).item()
    print("test loss: ", step, 
            ", loss: ", test_loss / (test_batch + 1), 
            ", loss_comp: ", test_loss_comp / (test_batch + 1), 
            
            ", sure_degree: ", sure_degree / (test_batch + 1), 
            ", lr: ", lr,
            # ", mean: ", mean, 
            # ", label: ", label, 
            )
    print(mean[: 2])
    print(label[: 2])
    # print(var[: 2])
    # print(probs[: 2])
    
    model.train()
    for _ in range(100):
        for item in dataloader:
            # print(item)
            
            step += 1
            predict_result = model.forward(item[0], item[1])
            
            # predict_result = model.forward(torch.concat([item[0], test_batch_restore[0]], 0), torch.concat([item[1], test_batch_restore[1]], 0))
            # print("0", item[0].isnan().sum())
            # print("1", item[1].isnan().sum())
            mean = predict_result[:, :7]
            var = predict_result[:, 7:]
            label = item[2].cuda()
            if step < 10000000:
                loss = torch.mean((mean - label) ** 2)
                loss +=  torch.mean(torch.abs(mean - label))
            else:
                probs = torch.exp(-((mean - label) / var) ** 2) / var
                probs[probs < 1e-10] = 1e-10
                loss = torch.mean(-1 * torch.log(probs))
            
            mean_diff = torch.mean(torch.abs(mean[0] - torch.mean(mean, dim=0)))
            mean_diff_mean = mean_diff_mean * 0.99 + mean_diff.item() * 0.01

            train_data_mean = 0.99 * train_data_mean + 0.01 * torch.mean(label, dim=0)
            train_loss_comp = 0.999 * train_loss_comp + 0.001 * (torch.mean((train_data_mean - label) ** 2) + torch.mean(torch.abs(train_data_mean - label)))
            
            if step < 100:
                mean_loss = loss.item()
                mean_var = torch.mean(var).item()
            else: 
                mean_loss = 0.999 * mean_loss + 0.001 * loss.item()
                mean_var = 0.999 * mean_var + 0.001 * torch.mean(var).item()
            
            if step % 500 == 0:
                torch.save(model.state_dict(), "model.pt")
            
            loss *= amp
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            # # stat nan
            # all_nan_num = 0
            # for item2 in model.parameters():
            #     all_nan_num += item2.isnan().sum()
            #     print(item2.isnan().sum(), torch.max(torch.abs(item2)))

            # 学习率过小会导致无法收敛，手动增大学习率
            if mean_diff_mean < 4e-4:
                amp *= 1.01
                print("amp: ", amp)
            if amp > 5:
                amp = 5
            if amp > 1:
                amp /= 1.0001
            
            
            if step % 10 == 0 or step < 100:
                # model.eval()
                # predict_result = model.forward(item[0], item[1])
                # mean2 = predict_result[:, :7]
                print("train loss: ", step, " steps", 
                      ", mean_loss: ", mean_loss, 
                      ", var: ", torch.mean(var).item(), 
                      ", mean_diff_mean", mean_diff_mean,
                      ", train_loss_comp ", train_loss_comp.item(),
                      
                      )
                # print(mean[0] == mean[1])
                print(mean[: 2])
                print(label[: 2])
                print(train_data_mean)
                # print(var[: 2])
                # print(mean2[: 2])
                # model.train()

    
        
        # for item2 in model.parameters():
        #     print(item2.isnan().sum())        

    
    
    
    
    