# Time series generation using GANs

This repository contains the implementation of a GAN-based method for generation of synthetic time series. This work has been done for exercise and it is inspired by https://github.com/proceduralia/pytorch-GAN-timeseries.


## Code structure
- `models/` : directory containing the model architecture for both discriminator and generator 
- `tsgan.py` : main class defining GAN training and sampling function for data synthesis. Generation can be unconditioned or conditioned on the difference between the last and the first element of the time series to be generated (e.g. a daily delta)
- `ts_dataset.py` : data loader class. Input dataset can have a column with the timestamp and multiple columns for features. User has to specify the names of the columns containing the timestamp and feature of interest and the time window to be used for training and generation - the number of timestamps to create a sequence (e.g. if dataset is a daily time series, she could consider 30 days sequences). Minimal preprocessing is perfomed including normalisation in the range [-1, 1]. The class has been designed to deal with time series exported e.g. from https://finance.yahoo.com/
- `test_run.py` : macro to run the training and sampling of synthetic time series. Can save the model checkpoints and images of generated time series, features visualisations (loss, gradients) via tensorboard
- `tsgan_configuration.yml` : configuration file containig settings for the GAN
- `read_sampled_data.ipynb` : simple notebook to load synthetic dataset and display generated timeseries

## Example
- set GAN configuration in tsgan_configuration
- run ``` python test_run.py ``` to train the network and generation of synthetic time series
- use ``` tensorboard --logdir log ``` from inside the project directory to run tensoboard on the default port (6006)
- use read_sampled_data.ipynb to visualise generated time series

During training, model weights are saved into the `checkpoints/` directory, snapshots of generated series into `images/` and tensorboard logs into `log/`.


## Interesting paper:
- [Real-valued (Medical) Time Series Generation with Recurrent Conditional GANs](https://arxiv.org/abs/1706.02633)



