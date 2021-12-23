import metrics.metrics
import numpy as np
import pandas as pd


filename = 'test_CNN_dis_LSTM_gen.npy'

X = np.load(filename, mmap_mode='r')

data = pd.DataFrame(X.reshape(4,30).T)

p = data.plot.line()