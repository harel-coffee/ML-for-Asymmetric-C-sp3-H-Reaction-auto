#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df1 = pd.read_csv('Set-1.csv')  # Put the new experiment file name here 


# In[ ]:


import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
from tqdm import tqdm_notebook
import time
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler    
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from keras.layers import Dropout
import csv
import concurrent.futures
import time
import itertools



df=pd.read_csv('MLS-LA-LB-LC-LD-Real-Synthetic80svm.csv') #add (real plus synthetic) datafile name here

#separate the real and synthetic dataset
Real =df.iloc[:240, :]
Synthetic =df.iloc[240:, :]


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)      
        self.regressor = nn.Sequential(
        nn.Linear(153, 33),
        nn.ReLU(),
        nn.Linear(33,150),
        nn.ReLU(),
        nn.Linear(150,400),
        nn.ReLU(),
        nn.Linear(400,168),
        nn.ReLU(),
        nn.Linear(168,128),
        nn.ReLU(),
        nn.Linear(128, 1)

        )

    def forward(self, x):
        x = self.regressor(x)
        return x



def multNNoob(seed, df1, df2, oob1):
    df_concat=pd.concat([df1, df2], axis=0)
    feature=df_concat.iloc[:, :-1].values
    output=df_concat.iloc[:, -1].values
    
    X_test1=oob1.iloc[:, :-1].values
    y_test1=oob1.iloc[:, -1].values


    

    X_train, X_val, y_train, y_val=train_test_split(feature, output, test_size=0.2, random_state=0)
    #make all the tensors
    X_train=torch.FloatTensor(X_train)
    X_val=torch.FloatTensor(X_val)
    y_train=torch.FloatTensor(y_train)
    y_val=torch.FloatTensor(y_val)

    X_test1=torch.FloatTensor(X_test1)
    y_test1=torch.FloatTensor(y_test1)



    #Instantiate the model
    model=NN()
    criterion=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    epoch=1000
    loss_arr=[]
    loss_val_arr=[]
    for epoch in range(epoch):
        optimizer.zero_grad()
        #Forward pass
        y_pred=model(X_train)
        #compute loss
        loss=criterion(y_pred.squeeze(), y_train)
        loss_arr.append(loss.item())
        #print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        #Backward pass
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            model.eval()
            y_pred_val=model(X_val)
            loss_val=criterion(y_pred_val.squeeze(), y_val)
            loss_val_arr.append(loss_val.item())
            #print('Epoch {}: val loss: {}'.format(epoch, loss_val.item()))
    
    #Now evaluate the test set1

    y_test1_predicted=[]
    y_test1_actual=[]
    model.eval()
    y_pred1 = model(X_test1)
    after_train1 = criterion(y_pred1.squeeze(), y_test1) 
    #print(torch.sqrt(after_train))
    test1_rmse=sqrt(after_train1)
    y_test1_predicted.append(y_pred1.tolist())
    y_test1_actual.append(y_test1.tolist())
    print(y_test1_predicted)



    return sqrt(loss_arr[-1]), sqrt(loss_val_arr[-1]), test1_rmse
    

result_oob=[]
for i in range(0, 1):
    result_oob.append(multNNoob(i, Real, Synthetic, df1))

dfResultNormal=pd.DataFrame(result_oob, columns=['train_rmse', 'val_rmse', 'new_exp_rmse'])
dfResultNormal.describe()

