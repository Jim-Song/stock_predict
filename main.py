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
import json



def train(args):
    # 设置训练 & 测试数据时间
    train_data_start_from = args.train_data_start_from # 1200
    train_data_end_at = args.train_data_end_at # 3000
    test_data_start_from = args.test_data_start_from # 3000
    test_data_end_at = args.test_data_end_at # 3499

    # 载入数据
    prices_dir = args.prices_dir # "datas_clean/prices"
    indexes_dir = args.indexes_dir # "datas_clean/indexes"
    prices = {}
    for file in os.listdir(prices_dir):
        tmp = pd.read_pickle(os.path.join(prices_dir, file))
        if tmp.open.isnull().sum() <= train_data_start_from and tmp.iloc[-1].isnull().sum() == 0 and len(tmp) >= 3499:
            prices[file[: 11]] = tmp
    indexes = {}
    for file in os.listdir(indexes_dir):
        tmp = pd.read_pickle(os.path.join(indexes_dir, file))
        if tmp.open.isnull().sum() <= train_data_start_from and tmp.iloc[-1].isnull().sum() == 0 and len(tmp) >= 3499:
            indexes[file[: 11]] = tmp
    prices_and_indexes = copy.copy(prices)
    prices_and_indexes.update(indexes)
    
    # 给予每只股票唯一的编号，训练时对应embedding不因股票数量变化发生改变
    if not os.path.exists("embedding_config.json"):
        os.system("echo \"{}\" > embedding_config.json")
    
    with open("embedding_config.json", "r") as f:
        embedding_config = json.load(f)
    
    # 已记录股票编号到第几
    order_to = 0
    for stock_code, order_num in embedding_config.items():
        if order_num > order_to:
            order_to = order_num
    
    for stock_code in prices_and_indexes.keys():
        if stock_code not in embedding_config:
            embedding_config[stock_code] = order_to
            order_to += 1

    with open("embedding_config.json", "w") as f:
        json.dump(embedding_config, f)
    
    
    
    
    

    # label_keys = indexes.keys()
    # for key in indexes.keys():
    #     print(key, indexes[key].open.isnull().sum())
    label_keys = ["000133.XSHE"]

    stocks_dataset = StocksDataset(prices, 
                                    indexes,
                                    label_name=label_keys,
                                    data_start_from=train_data_start_from, 
                                    data_end_at=train_data_end_at, 
                                    num1=args.train_stock_num, # 1200
                                    use_rand=True,
                                    embedding_config=embedding_config,
                                    )

    stocks_dataset_test = StocksDataset(prices, 
                                        indexes,
                                        label_name=label_keys,
                                        data_start_from=test_data_start_from, 
                                        data_end_at=test_data_end_at,
                                        num1=-1, 
                                        use_rand=False,
                                        embedding_config=embedding_config,
                                        only_test=args.only_test,
                                        only_need_result=args.only_need_result,
                                        )
    # 训练集 & 测试集
    dataloader = DataLoader(stocks_dataset, batch_size=args.batch_size, shuffle=True, num_workers=25)
    dataloader2 = DataLoader(stocks_dataset_test, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device('cuda:0') if args.use_cuda else torch.device('cpu')
    model = Model().to(device)
    if args.restore:
        map_location = torch.device('cuda:0') if args.use_cuda else torch.device('cpu')
        model.load_state_dict(torch.load(args.restore, map_location=map_location))

    lr = 5e-5
    step = 0
    mean_loss = 0
    mean_var = 0
    mean_diff_mean = 6e-4

    train_loss_comp = 1.0
    train_data_mean = torch.tensor([0.0] * 7).to(device)

    # 前 train_var_step 步只训练mean，不训练var
    train_var_step = 50000

    while True:
        test_loss = 0
        test_loss_comp = 0
        sure_degree = 0
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
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
            mean_npy = mean.detach().cpu().numpy().copy()
            label_npy = label.detach().cpu().numpy().copy()
            var_npy = var.detach().cpu().numpy().copy()
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
            
            # large_mean_indexes = np.where(mean_npy[0, :] == mean_npy[0, :].max(axis=0))
            # large_mean_indexes2 = list(zip(large_mean_indexes[0], large_mean_indexes[1]))
            # large_mean_indexes2.sort(key=lambda x: x[1])
            for adv_i in range(7):
                # 取出每个持股期限上预测 mean 最大 test_stock_num 只股票的 index
                hold_stock_indexes = []
                current_advs = []
                current_adv_avg = 0
                if adv_i == 0:
                    test_stock_num = args.test_stock_num * 10
                # else:
                #     test_stock_num = args.test_stock_num
                for stock_num_idx in range(test_stock_num):
                    choose_stock = False
                    for _ in range(100):
                        large_mean_index = np.where(mean_npy[0, :, adv_i] == mean_npy[0, :, adv_i].max(axis=0))[0][0]
                        mean_npy[0, :, adv_i][large_mean_index] = mean_npy[0, :, adv_i].min() - 1
                        # 获取股票代码
                        stock_index = stocks_dataset_test.bianhao_code_dict[large_mean_index]
                        # 只买流动性好的股票
                        # print("当前股票： ", stock_index, ", 成交额： ", prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx].money)
                        if prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx].money > 1e8 \
                            and prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx].close > 3.0:
                            choose_stock = True
                            break
                        # print("流动性不足，忽略： ", stock_index, ", 成交额： ", prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx].money)
                    if not choose_stock:
                        print("no match")
                        break
                        
                    hold_stock_indexes.append(large_mean_index)
                    current_adv = label_npy[0, large_mean_index, adv_i]
                    current_advs.append(current_adv)
                    
                    current_price = -1
                    future_price = -1
                    
                    hold_stock_days = int(np.floor((stocks_dataset_test.time_slices_label[adv_i + 1] + stocks_dataset_test.time_slices_label[adv_i + 2]) / 2)) - 1
                    if test_data_start_from + test_idx + hold_stock_days + 1 >= prices_and_indexes[stock_index].shape[0]:
                        print("| code: ", stock_index, 
                                "\t 持有天数: ", hold_stock_days, 
                                )
                        continue
                    current_price = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx + 1].open
                    

                    if test_data_start_from + test_idx + hold_stock_days + 1 < min(test_data_end_at, len(prices_and_indexes[stock_index])):
                        future_price = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx + hold_stock_days + 1].open
                        sell_stock_date = prices_and_indexes[stock_index].iloc[test_data_start_from + test_idx + hold_stock_days + 1].name
                        
                    if current_price > 0 and future_price > 0:
                        ratio_tmp = future_price / current_price
                        # 不计算复利，每一时刻拿出本金的 1 / args.test_stock_num / hold_stock_days 购买新股票
                        # 到期卖出后，如果股票价值低于本金的 1 / args.test_stock_num / hold_stock_days 则将本金补至初始本金并在利润中减去亏损值
                        # 到期卖出后，如果股票价值高于本金的 1 / args.test_stock_num / hold_stock_days 则在总利润中加上当前利润
                        ratios[adv_i] = ratios[adv_i] + (ratio_tmp - 1) / test_stock_num / hold_stock_days
                        if stock_num_idx < args.test_stock_num:
                            print("| code: ", stock_index, 
                                "\t 持有天数: ", hold_stock_days, 
                                "\t 净值:", str(ratios[adv_i])[: 5], 
                                "\t 售价/现价:", str(ratio_tmp)[: 5], 
                                "\t 现价:", str(current_price)[: 5], 
                                "\t 售价:", str(future_price)[: 5] , 
                                "\t 出售日期:", sell_stock_date , 
                                "\t 优势度:", str(current_adv)[: 5] , 
                                "\t|",
                                )
                daily_ratios[adv_i].append(ratios[adv_i])
                    
                # # 获取股票的index和持有期限
                # # hold_stock_days_index = adv_i
                # hold_stock_index = large_mean_indexes2[adv_i][0]
                # # current_adv = label_npy[0, hold_stock_index, hold_stock_days_index]
                # advs[adv_i].append(sum(current_advs) / len(current_advs))
                if len(current_advs) > 0:
                    current_adv_avg = sum(current_advs) / len(current_advs)
                else:
                    current_adv_avg = 1
                advs[adv_i].append(current_adv_avg)
            aaa = 1
        # print(advs)
        print("ratios: ", ratios)
        adv = 0
        adv_means = []
        for adv_i in range(7):
            adv = sum(advs[adv_i]) / len(advs[adv_i])
            adv_means.append(adv)
        # adv_mean = sum(advs2) / len(advs2)
        pullbacks = []
        
        for i in range(len(daily_ratios)):
            plt.plot(dates, daily_ratios[i])
            hold_stock_days = int(np.floor((stocks_dataset_test.time_slices_label[i + 1] + stocks_dataset_test.time_slices_label[i + 2]) / 2)) - 1
            plt.xticks(rotation=30)
            plt.savefig("持有天数" + str(hold_stock_days) + ".png")
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
        for _ in range(80):
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
                    torch.save(model.state_dict(), args.save_model)
                
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
        "--train_data_start_from",
        type=int,
        help="训练数据开始index",
        default=1200,
    )
    parser.add_argument(
        "--train_data_end_at",
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
        default="",
    )
    parser.add_argument(
        "--save_model",
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
        "--test_stock_num",
        type=int,
        help="",
        default=1,
    )
    parser.add_argument(
        "--only_need_result",
        type=int,
        help="",
        default=0,
    )

    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    train(args)


    
    