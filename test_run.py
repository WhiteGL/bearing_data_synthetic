import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import datetime
from ts_dataset import TSDataset
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator
from tsgan import TSGANSynthetiser
import yaml

#load config file
path_to_yaml = 'tsgan_configuration.yml'
try: 
    with open(path_to_yaml, 'r') as file:
        config = yaml.safe_load(file)
except Exception:
    print('Error reading the config file')

#Create writer for tensorboard
date = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")
run_name = f"{config['run_tag']}_{date}" if config['run_tag'] != '' else date
#log_dir_name = os.path.join(config['logdir'], run_name)
log_dir_name = config['logdir']
writer = SummaryWriter(log_dir_name)

try:
    os.makedirs(config['outf'])
except OSError:
    pass
try:
    os.makedirs(config['imf'])
except OSError:
    pass


fit = config['fit']
sample_new_data = config['sample_new_data']

tsgan = TSGANSynthetiser(path_to_yaml, writer)
if fit:
    print('****************************** Fitting ******************************')
    tsgan.fit()
elif sample_new_data:
    print('****************************** Sampling ******************************')
    sample = tsgan.sample_data()
else:
    print('****************************** No valid option selected ******************************')