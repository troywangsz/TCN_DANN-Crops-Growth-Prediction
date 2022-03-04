import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Function
import torch.nn.functional as F
from math import pi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataset import plant_dataset
from model_DANN_GaussianTCN import DANN_GaussianTCN
import argparse
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--cuda', type=int, default=0)
parser.add_argument('--nepoch', type=int, default=20000)
parser.add_argument('--layer', type=int, default=4)
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':
    # torch.manual_seed(100004)

    source_data = pd.read_csv("data/source_data_maize.csv").values
    target_data = pd.read_csv("data/target_data_maize.csv").values


    source_data = source_data.reshape((-1,11,6))
    source_dataset = plant_dataset(source_data)
    target_data = target_data.reshape((-1,11,6))
    target_train_data, target_valid_data,target_test_data= torch.utils.data.random_split(
        dataset=target_data, lengths=[32, 8, 6], generator=torch.Generator().manual_seed(509)
    )

    target_train_dataset = plant_dataset(target_train_data)
    target_valid_dataset = plant_dataset(target_valid_data)
    target_test_dataset = plant_dataset(target_test_data)

    source_dataloader = torch.utils.data.DataLoader(dataset=source_dataset,batch_size=args.batchsize,shuffle=True)
    target_train_dataloader = torch.utils.data.DataLoader(dataset=target_train_dataset,batch_size=args.batchsize,shuffle=True)
    target_valid_dataloader = torch.utils.data.DataLoader(dataset=target_valid_dataset,batch_size=args.batchsize,shuffle=True)
    target_test_dataloader = torch.utils.data.DataLoader(dataset=target_test_dataset,batch_size=6,shuffle=True)

    len_dataloader = min(len(source_dataloader), len(target_train_dataloader))
    model = DANN_GaussianTCN(feature_layer=[40]*args.layer)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion_mae = torch.nn.L1Loss()
    best_loss = 100



    for epoch in range(args.nepoch):
        model.train()
        # torch.autograd.set_detect_anomaly(True)
        for ((i,(source_x,source_y)),(_,(target_x,target_y))) in zip(enumerate(source_dataloader),enumerate(target_train_dataloader)):
            source_x = source_x.to(DEVICE)
            source_y = source_y.to(DEVICE)
            target_x = target_x.to(DEVICE)
            target_y = target_y.to(DEVICE)

            p = float(epoch * len_dataloader + i + 1) / (args.nepoch * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            source_regression_output,source_domain_output = model(input=source_x,alpha=alpha)

            source_regression_loss,source_domain_loss = model.loss(
                regression_output=source_regression_output,
                domain_output=source_domain_output,
                regression_label=source_y,
                device=DEVICE,
                source=True
            )

            target_regression_output,target_domain_output = model(input=target_x,alpha=alpha)
            target_regression_loss,target_domain_loss = model.loss(
                regression_output=target_regression_output,
                domain_output=target_domain_output,
                regression_label=target_y,
                device=DEVICE,
                source=False
            )

            total_loss = source_regression_loss + args.gamma * (source_domain_loss + target_domain_loss)
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            total_loss.backward()
            optimizer.step()


        # item_pr = 'Epoch: [{}/{}]\nsource_regression_loss: {:.4f}\nsource_domain_loss: {:.4f}\ntarget_domain_loss: {:.4f}\ntotal_loss: {:.4f}\n'.format(
        #     epoch, args.nepoch, source_regression_loss.item(), source_domain_loss.item(), target_domain_loss.item(),
        #     total_loss.item())
        # print(item_pr)
        print("Epoch[%d/%d]" % (epoch + 1, args.nepoch))
        print("Train Loss:%f"%total_loss.item())

        # valid
        valid_loss = 0
        model.eval()
        valid_num = 0
        with torch.no_grad():
            for i,(valid_x,valid_y) in enumerate(target_valid_dataloader):
                valid_x = valid_x.to(DEVICE)
                valid_y = valid_y.to(DEVICE)
                valid_regression_output,valid_domain_output = model(input=valid_x,alpha=0)


                # valid_regression_loss,valid_domain_loss = model.loss(
                #     regression_output=valid_regression_output,
                #     domain_output=torch.zeros(args.batchsize).float(),
                #     regression_label=valid_y,
                #     source=False
                # )
                valid_regression_loss = torch.sqrt(torch.mean((valid_regression_output[0]-valid_y)**2))

                valid_loss += valid_regression_loss
                valid_num += 1
            valid_loss = valid_loss.item() / valid_num
            print("Valid Loss:%f" % valid_loss)


        with torch.no_grad():
            for i,(test_x,test_y) in enumerate(target_test_dataloader):
                test_x = test_x.to(DEVICE)
                test_y = test_y.to(DEVICE)
                test_regression_output,test_domain_output = model(input=test_x,alpha=0)


                # valid_regression_loss,valid_domain_loss = model.loss(
                #     regression_output=valid_regression_output,
                #     domain_output=torch.zeros(args.batchsize).float(),
                #     regression_label=valid_y,
                #     source=False
                # )
                test_regression_loss_rmse = torch.sqrt(torch.mean((test_regression_output[0] - test_y) ** 2))
                test_regression_loss_mae = criterion_mae(test_regression_output[0], test_y)
            print("Test RMSE Loss:%f" % test_regression_loss_rmse.item())
            print("Test MAE Loss:%f" % test_regression_loss_mae.item())


        # if valid_loss < best_loss:
        #     best_loss = valid_loss
        #     print("new best valid loss: %f" % best_loss)
        #     torch.save(model, "result/best_TCN_model_param_maize")
        # print(best_loss)


