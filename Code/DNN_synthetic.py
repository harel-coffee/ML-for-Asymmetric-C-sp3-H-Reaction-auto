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



df=pd.read_csv('MLS-LA-LB-LC-LD-Real-Synthetic.csv') #add (real plus synthetic) datafile name here

#separate the real and synthetic dataset
Real =df.iloc[:240, :]
Synthetic =df.iloc[240:, :]




class NN(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.regressor = nn.Sequential(
            nn.Linear(153, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, 1)

        )

    def forward(self, x):
        x = self.regressor(x)
        return x



def multNN1(seed, df1, df2):
    exp_feat=df1.iloc[:, :-1]
    exp_ee=df1.iloc[:, -1]
    X_exp_train, X_test, y_exp_train, y_test = train_test_split(exp_feat, exp_ee, test_size=0.2, random_state=seed)
    print(X_test.index)
    ind_val=X_test.index
    df_except_test=pd.concat([X_exp_train, y_exp_train], axis=1)
    df_real_syn=pd.concat([df_except_test, df2], axis=0)
    #now separate train and validation set from the combined data i.e. df_real_syn
    X_train_val=df_real_syn.iloc[:, :-1].values
    y_train_val=df_real_syn.iloc[:, -1].values
    X_train, X_val, y_train, y_val=train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=0)
    #make all the tensors
    X_train=torch.FloatTensor(X_train)
    X_val=torch.FloatTensor(X_val)
    y_train=torch.FloatTensor(y_train)
    y_val=torch.FloatTensor(y_val)
    X_test=torch.FloatTensor(X_test.values)
    y_test=torch.FloatTensor(y_test.values)
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
    
    #Now evaluate the test set
    model.eval()
    y_pred = model(X_test)
    after_train = criterion(y_pred.squeeze(), y_test) 
    #print(torch.sqrt(after_train))
    test_rmse=sqrt(after_train)
    list1= ind_val
    with open('MLS-LA-LB-LC-LD-Real-Synthetic-out.csv', 'a') as csvFile:
         writer = csv.writer(csvFile)
         writer.writerows(zip(itertools.repeat(seed),list1, y_test, y_pred))
    csvFile.close()

    return seed, sqrt(loss_arr[-1]), sqrt(loss_val_arr[-1]), test_rmse



result=[]
for i in range(0, 100):
    result.append(multNN1(i,Real, Synthetic))


dfResultNormal=pd.DataFrame(result, columns=['seed', 'train_rmse', 'val_rmse', 'test_rmse'])

dfResultNormal.describe()
dfResultNormal.to_csv('MLS-LA-LB-LC-LD-Real-Synthetic-Result.csv')


print(dfResultNormal.describe())