import datetime
import yaml
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from metrics import numerical_metrics
from metrics.visualization_metrics import PCA_Analysis, tSNE_Analysis

filename = 'test_CNN_dis_LSTM_gen20_34.npy'
filename_true = './data/Norm1.csv'
window = 400

Y = pd.read_csv(filename_true)
y = np.array(Y['DE'])
data_true = np.zeros((608, 400))
for i in range(608):
    data_true[i] = y[i*400:(i+1)*400]

X = np.load(filename, mmap_mode='r')
data = np.array(X.reshape((608, 400)))
PCA_Analysis(data_true, data)