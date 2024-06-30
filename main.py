import torch
from load_data import StocksDataset
from model import Model
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt



def train(args):
    # 设置训练 & 测试数据时间
    data_start_from = args.data_start_from # 1200
    data_end_at = args.data_end_at # 3000
    test_data_start_from = args.test_data_start_from # 3000
    test_data_end_at = args.test_data_end_at # 3499

    # 载入数据
    prices_dir = args.prices_dir # "datas_clean/prices"
    indexes_dir = args.indexes_dir # "datas_clean/indexes"
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
    prices_and_indexes = copy.copy(prices)
    prices_and_indexes.update(indexes)

    # label_keys = indexes.keys()
    # for key in indexes.keys():
    #     print(key, indexes[key].open.isnull().sum())
    label_keys = ["000133.XSHE"]

    stocks_dataset = StocksDataset(prices, 
                                    indexes,
                                    label_name=label_keys,
                                    data_start_from=data_start_from, 
                                    data_end_at=data_end_at, 
                                    num1=args.train_stock_num, # 1200
                                    use_rand=True,
                                    )

    stocks_dataset_test = StocksDataset(prices, 
                                        indexes,
                                        label_name=label_keys,
                                        data_start_from=test_data_start_from, 
                                        data_end_at=test_data_end_at,
                                        num1=2847, 
                                        use_rand=False,
                                        )
    # 训练集 & 测试集
    dataloader = DataLoader(stocks_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader2 = DataLoader(stocks_dataset_test, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device('cuda:0') if args.use_cuda else torch.device('cpu')
    model = Model().to(device)
    if args.restore:
        map_location = torch.device('cuda:0') if args.use_cuda else torch.device('cpu')
        model.load_state_dict(torch.load(args.restore, map_location=map_location))

    lr = 2e-4
    step = 0
    mean_loss = 0
    mean_var = 0
    mean_diff_mean = 4e-4

    train_loss_comp = 1.0
    train_data_mean = torch.tensor([0.0] * 7).to(device)

    # 前 train_var_step 步只训练mean，不训练var
    train_var_step = 50000

    while True:
        test_loss = 0
        test_loss_comp = 0
        sure_degree = 0
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        lr *= 0.98
        
        model.eval()
        test_batch_restore = None
        # 策略的 advantage，共7种策略
        advs = [[], [], [], [], [], [], [], ]
        ratios = [1, 1, 1, 1, 1, 1, 1]
        daily_ratios = [[], [], [], [], [], [], [], ]
        dates = []
        # advs2 = []
        for test_batch, item in enumerate(dataloader2):
            if not test_batch_restore:
                test_batch_restore = item
            # print(step)
            # print(item)
            # if item[0].shape[0] <= 1:
            #     continue
            mean, var = model.forward(item[0].to(device), item[1].to(device))
            # mean = predict_result[:, :, :7]
            # var = predict_result[:, 7:]
            label = item[2].to(device)
            
            probs = torch.exp(-((mean - label) / var) ** 2) / var
            probs[probs < 1e-5] = 1e-5
            loss = torch.mean(-1 * torch.log(probs))
            
            # loss = torch.mean(torch.abs((mean - label)))
            zeros = torch.zeros(mean.shape).to(device)

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
            
            
            # large_mean_indexes = np.where(mean_npy[0, :, 0] == mean_npy[0, :, 0].max())
            # adv = label_npy[0, :, 0][large_mean_indexes].mean()
            # if not np.isnan(adv):
            #     advs2.append(adv)
                
            print("=====================================================================================================================")
            test_idx = item[3].item() - stocks_dataset_test.time_slices[-1]
            date_ = indexes["000001.XSHG"].iloc[test_data_start_from + test_idx].name
            dates.append(date_)
            print("|\t\t\t时间: ", date_, "\t\t\t|")
            # 取出每个持股期限上预测mean最大的股票index
            large_mean_indexes = np.where(mean_npy[0, :] == mean_npy[0, :].max(axis=0))
            large_mean_indexes2 = list(zip(large_mean_indexes[0], large_mean_indexes[1]))
            large_mean_indexes2.sort(key=lambda x: x[1])
            for adv_i in range(7):
                # 获取股票的index和持有期限
                hold_stock_days_index = large_mean_indexes2[adv_i][1]
                hold_stock_index = large_mean_indexes2[adv_i][0]
                current_adv = label_npy[0, hold_stock_index, hold_stock_days_index]
                advs[hold_stock_days_index].append(current_adv)
                
                # 获取股票代码
                stock_index = stocks_dataset_test.key_indexes_dict[hold_stock_index]
                current_price = -1
                future_price = -1
                
                hold_stock_days = stocks_dataset_test.time_slices_label[hold_stock_days_index + 2]
                current_price = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx].close
                if test_data_start_from + test_idx + hold_stock_days < test_data_end_at:
                    future_price = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx + hold_stock_days].close
                    sell_stock_date = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx + hold_stock_days].name
                    
                if current_price > 0 and future_price > 0:
                    ratio_tmp = future_price / current_price
                    ratios[adv_i] *= ratio_tmp
                    daily_ratio = ratios[adv_i] ** (1 / hold_stock_days)
                    print("| 股票: ", stock_index, 
                          "\t 持股天数: ", hold_stock_days, 
                          "\t 策略净值:", str(daily_ratio)[: 5], 
                          "\t 股票售价/现价:", str(ratio_tmp)[: 5], 
                          "\t 现价:", str(current_price)[: 5], 
                          "\t 售价:", str(future_price)[: 5] , 
                          "\t 出售日期:", sell_stock_date , 
                          "\t 优势度:", str(current_adv)[: 5] , 
                          "\t|",
                          )
                    daily_ratios[adv_i].append(daily_ratio)
            aaa = 1

        # print(advs)
        print(ratios)
        adv = 0
        adv_means = []
        for adv_i in range(7):
            adv = sum(advs[adv_i]) / len(advs[adv_i])
            adv_means.append(adv)
        # adv_mean = sum(advs2) / len(advs2)
        pullbacks = []
        
        for i in range(len(daily_ratios)):
            plt.plot(dates, daily_ratios[i])
            plt.savefig("持股天数" + str(stocks_dataset_test.time_slices_label[i + 2]) + ".png")
            plt.close()
            max_pullback = 1
            for j in range(len(daily_ratios[i])):
                for k in range(j):
                    ratio_tmp = daily_ratios[i][j] / daily_ratios[i][k]
                    if ratio_tmp < max_pullback:
                        max_pullback = ratio_tmp
            max_pullback = 1 - max_pullback
            pullbacks.append(max_pullback)
        print("pullbacks: ", pullbacks)

        print("test loss: ", step, 
                ", loss: ", test_loss / (test_batch + 1), 
                ", loss_comp: ", test_loss_comp / (test_batch + 1), 
                ", sure_degree: ", sure_degree / (test_batch + 1), 
                ", lr: ", lr,
                ", probs < 1e-4: ", (probs < 1e-4).sum(),
                ", adv: ", adv_means,
                # ", adv_mean: ", adv_mean,
                # ", mean: ", mean, 
                # ", label: ", label, 
                )
        print(mean[: 2, 0])
        print(label[: 2, 0])
        print(var[: 2, 0])
        print(probs[: 2, 0])
        # time.sleep(100)
        if args.only_test:
            break
        
        model.train()
        for _ in range(2):
            for item in dataloader:
                # print(item)
                step += 1
                mean, var = model.forward(item[0].to(device), item[1].to(device))
                # predict_result = model.forward(torch.concat([item[0], test_batch_restore[0]], 0), torch.concat([item[1], test_batch_restore[1]], 0))
                # print("0", item[0].isnan().sum())
                # print("1", item[1].isnan().sum())
                # mean = predict_result[:, :, :7]
                # var = predict_result[:, 7:]
                label = item[2].to(device)
                if step < train_var_step:
                    loss = torch.mean((mean - label) ** 2)
                    loss +=  torch.mean(torch.abs(mean - label))
                else:
                    probs = torch.exp(-((mean - label) / var) ** 2) / var
                    # 异常值不训练，避免过大梯度
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
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # # stat nan
                # all_nan_num = 0
                # for item2 in model.parameters():
                #     all_nan_num += item2.isnan().sum()
                #     print(item2.isnan().sum(), torch.max(torch.abs(item2)))
                
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prices_dir",
        type=str,
        help="data1",
        default="datas_clean/prices",
    )
    parser.add_argument(
        "--indexes_dir",
        type=str,
        help="data2",
        default="datas_clean/indexes",
    )
    parser.add_argument(
        "--data_start_from",
        type=int,
        help="训练数据开始index",
        default=1200,
    )
    parser.add_argument(
        "--data_end_at",
        type=int,
        help="训练数据结束index",
        default=3000,
    )
    parser.add_argument(
        "--test_data_start_from",
        type=int,
        help="测试数据开始index",
        default=3000,
    )
    parser.add_argument(
        "--test_data_end_at",
        type=int,
        help="测试数据结束index",
        default=3499,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch_size",
        default=3,
    )
    parser.add_argument(
        "--use_cuda",
        type=int,
        help="",
        default=1,
    )
    parser.add_argument(
        "--restore",
        type=str,
        help="",
        default="model.pt",
    )
    parser.add_argument(
        "--train_stock_num",
        type=int,
        help="",
        default=1200,
    )
    parser.add_argument(
        "--only_test",
        type=int,
        help="",
        default=0,
    )
    parser.add_argument(
        "--time_interval",
        type=int,
        help="",
        default=-1,
    )
    parser.add_argument(
        "--get_model_from_exploiter_time_interval",
        type=int,
        help="",
        default=36000,
    )
    parser.add_argument(
        "--stat_dirpath",
        type=str,
        help="",
        default=36000,
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train(args)


    
    