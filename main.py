import torch
from load_data import StocksDataset
from model import Model
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse


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
                               num1=1200, 
                               use_rand=True,
                               )

stocks_dataset_test = StocksDataset(prices, 
                                    indexes,
                                    label_name=label_keys,
                                    data_start_from=test_data_start_from, 
                                    data_end_at=test_data_end_at,
                                    num1=2800, 
                                    use_rand=False,
                                    )

dataloader = DataLoader(stocks_dataset, batch_size=4, shuffle=True, num_workers=4)
dataloader2 = DataLoader(stocks_dataset_test, batch_size=1, shuffle=False, num_workers=5)


model = Model().cuda()
model.load_state_dict(torch.load("model.pt"))

lr = 2e-4

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

train_loss_comp = 1.0
train_data_mean = torch.tensor([0.0] * 7).cuda()

train_var_step = 1

while True:
    test_loss = 0
    test_loss_comp = 0
    sure_degree = 0
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    lr *= 0.98
    
    model.eval()
    test_batch_restore = None
    advs = [[], [], [], [], [], [], [], ]
    advs2 = []
    for test_batch, item in enumerate(dataloader2):
        if not test_batch_restore:
            test_batch_restore = item
        # print(step)
        # print(item)
        # if item[0].shape[0] <= 1:
        #     continue
        mean, var = model.forward(item[0], item[1])
        # mean = predict_result[:, :, :7]
        # var = predict_result[:, 7:]
        label = item[2].cuda()
        
        probs = torch.exp(-((mean - label) / var) ** 2) / var
        probs[probs < 1e-5] = 1e-5
        loss = torch.mean(-1 * torch.log(probs))
        
        # loss = torch.mean(torch.abs((mean - label)))
        zeros = torch.zeros(mean.shape).cuda()

        probs_comp = torch.exp(-((zeros - label) / var) ** 2) / var
        probs_comp[probs_comp < 1e-5] = 1e-5
        loss_comp = torch.mean(-1 * torch.log(probs_comp)).detach().item()
        test_loss_comp += loss_comp
        
        test_loss += loss.detach().item()
        sure_degree += torch.mean(var).item()
        mean_npy = mean.detach().cpu().numpy()
        label_npy = label.detach().cpu().numpy()
        var_npy = var.detach().cpu().numpy()
        # var_thres = 0.5
        # mean_thres = 0.5
        # little_var_indexes = np.where(var_npy[0, :, 0] < var_thres)
        # large_mean_indexes = np.where(mean_npy[0, :, 0][little_var_indexes] > mean_thres)
        # if large_mean_indexes[0].size == 0:
        #     large_mean_indexes = np.where(mean_npy[0, :, 0][little_var_indexes] == mean_npy[0, :, 0][little_var_indexes].max())
        # adv = label_npy[0, :, 0][little_var_indexes][large_mean_indexes].mean()
        
        
        large_mean_indexes = np.where(mean_npy[0, :, 0] == mean_npy[0, :, 0].max())
        adv = label_npy[0, :, 0][large_mean_indexes].mean()
        if not np.isnan(adv):
            advs2.append(adv)
        
        large_mean_indexes = np.where(mean_npy[0, :] == mean_npy[0, :].max(axis=0))
        for adv_i in range(7):
            advs[large_mean_indexes[1][adv_i]].append(label_npy[0, large_mean_indexes[0][adv_i], large_mean_indexes[1][adv_i]])
        
        aaa = 1
# advs
# advs2

    print(advs)
    adv = 0
    adv_means = []
    for adv_i in range(7):
        adv = sum(advs[adv_i]) / len(advs[adv_i])
        adv_means.append(adv)
    
    adv_mean = sum(advs2) / len(advs2)
        
    print("test loss: ", step, 
            ", loss: ", test_loss / (test_batch + 1), 
            ", loss_comp: ", test_loss_comp / (test_batch + 1), 
            
            ", sure_degree: ", sure_degree / (test_batch + 1), 
            ", lr: ", lr,
            ", probs < 1e-4: ", (probs < 1e-4).sum(),
            ", adv: ", adv_means,
            ", adv_mean: ", adv_mean,
            # ", mean: ", mean, 
            # ", label: ", label, 
            )
    print(mean[: 2, 0])
    print(label[: 2, 0])
    print(var[: 2])
    print(probs[: 2])
    
    model.train()
    for _ in range(2):
        for item in dataloader:
            # print(item)
            
            step += 1
            mean, var = model.forward(item[0], item[1])
            
            # predict_result = model.forward(torch.concat([item[0], test_batch_restore[0]], 0), torch.concat([item[1], test_batch_restore[1]], 0))
            # print("0", item[0].isnan().sum())
            # print("1", item[1].isnan().sum())
            # mean = predict_result[:, :, :7]
            # var = predict_result[:, 7:]
            label = item[2].cuda()
            if step < train_var_step:
                loss = torch.mean((mean - label) ** 2)
                loss +=  torch.mean(torch.abs(mean - label))
            else:
                probs = torch.exp(-((mean - label) / var) ** 2) / var
                probs[probs < 1e-5] = 1e-5
                loss = torch.mean(-1 * torch.log(probs))
            
            mean_diff = torch.mean(torch.abs(mean[0] - torch.mean(mean, dim=0)))
            mean_diff_mean = mean_diff_mean * 0.99 + mean_diff.item() * 0.01

            train_data_mean = 0.99 * train_data_mean + 0.01 * torch.mean(torch.mean(label, dim=0), dim=0)
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
                print(mean[: 2, 0])
                print(label[: 2, 0])
                # print(train_data_mean)
                if step > train_var_step:
                    print(var[: 2, 0])
                    print(probs[: 2, 0])
                    print("probs < 1e-4 ", (probs < 1e-4).sum())
                    print("mean_var: ", mean_var)
                # print(var[: 2])
                # print(mean2[: 2])
                # model.train()

    
        
        # for item2 in model.parameters():
        #     print(item2.isnan().sum())        

    
    
    
    
    