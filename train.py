# Author: Xinwen Xu
# Date: 2022/8/1

import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from time import process_time
from sklearn.model_selection import train_test_split

import dataloader
import model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#load data
train_data, test_data = dataloader.load(device, partition=0.7)

n_train = len(train_data)
n_test = len(test_data)


# set model parameters
data_size, words_size, strokes_size, num_ids = dataloader.info()
b_h_size, s_h_size = 200, 60
inp_size = 80
hidden_size = 150

# Initialize a model
Elstms = model.LSTMs(words_size, strokes_size, b_h_size, s_h_size, inp_size, hidden_size, device).to(device)

# set training parameters
optimizer = torch.optim.Adam(Elstms.parameters(), lr = 1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.9) 
n_epoch = 5

#criterion = nn.MSELoss()
criterion = nn.L1Loss()

epoch_train_loss = np.empty(n_epoch)
epoch_test_loss = np.empty(n_epoch)
#id_train_loss = np.empty([n_epoch, num_ids])


def train_model():
    for epoch in range(n_epoch):
        t1 = process_time()

        random.shuffle(train_data)

        train_loss = 0  # total loss on the whole train data set
        test_loss = 0

        # train
        Elstms.train()

        for data in train_data:
            # skip null data, which are hard to discard in previous processing
            if len(data[0])==0:
                n_train -= 1
                continue
            
            Elstms.zero_grad()
            prediction = Elstms(data[0], data[1], data[2])
            target = torch.tensor([data[3]]).to(device)
            loss = criterion(prediction, target)
            #loss = criterion(prediction.item(),data[3])
            train_loss += loss
            loss.backward()
            optimizer.step()

        #scheduler.step()

        # test
        with torch.no_grad():
            for data in test_data:
                if len(data[0])==0:
                    n_test -= 1
                    continue
                
                prediction = Elstms(data[0], data[1], data[2])
                target = torch.tensor([data[3]]).to(device)
                loss = criterion(prediction, target)
                test_loss += loss

        epoch_train_loss[epoch] = train_loss/n_train
        epoch_test_loss[epoch] = test_loss/n_test

        t2 = process_time()

        print(f"Epoch{epoch}: average train loss={epoch_train_loss[epoch]}, average test loss={epoch_test_loss[epoch]}, \
            took time: {t2-t1} seconds.",
                flush=True)

    return Elstms, epoch_train_loss, epoch_test_loss

