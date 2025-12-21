# DSTN-RA: A Time Series Forecasting Model

This repository contains the implementation of DSTN-RA (Decomposed Seasonal-Trend Network with Rotary Attention), a novel time series forecasting model.

This is the official code and supplementary materials for LIMU's work on Time Series Forecasting.

# Datasets

We currently support the following datasets:

ETT (Electricity Transformer Temperature)

Exchange (Exchange Rate)

Weather

In the future, more datasets will be included.

## Setup Instructions:

Please unzip the `raw_data.rar` file into the root directory under the folder `./datasets`.

### Example directory structure after unzipping

`./datasets/`

` ├── ETT/`
  
` ├── exchange_rate/`
  
` └── weather/`

Note: More datasets will be released in the future.

# Model Architecture

The core framework of the model is located in `./model/DSTNRA.py`.

The architecture primarily consists of two key stages:

Decomposition: The input time series signals are first decomposed.

Processing Modules: The decomposed components are processed through Channel Mixing and Rotary Attention modules to capture temporal dependencies and feature interactions.

# Running

## Requirements

Please ensure you have installed the necessary library dependencies before running the code.

## Training

You can start training the model by running the `train.py`. You can modify them via command-line arguments.

Run `python train.py`

## Key Arguments

If you wish to customize the training process, please pay attention to the following arguments:

Prediction Length: Use `--pred_len` to change the forecasting horizon.

Dataset Configuration: When switching datasets, ensure you check the number of features in your target dataset and update the following arguments accordingly:

`--enc_in`: Encoder input size

`--dec_in`: Decoder input size

`--c_out`: Output size


# Future Plans

We are committed to improving this project. Following the acceptance of the paper, we plan to:

Integrate the Optuna framework to allow users to automatically fine-tune hyperparameters.

Provide the optimal hyperparameters for each supported dataset."# DSTN-RA_LIMU" 
