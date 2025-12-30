import argparse
import os
from exp_forecast import ExpForecast
import torch.multiprocessing
import numpy as np
import random
import warnings
import sys
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def parse_args():
    parser = argparse.ArgumentParser(description="Time Series Platform Args")
    parser.add_argument('--model', type=str, default='DSTNRA', help='*select the model you want to train')
    parser.add_argument('--optimizer', type=str, default='Adam', help='*select the optimizer you want to adopt')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='*select the initial learning rate')
    parser.add_argument('--train_epochs', type=int, default=100, help='*select the number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='*select the batch size for training')
    parser.add_argument('--use_gpu', default=True, help='*decide whether to use GPU or not')
    parser.add_argument('--gpu', type=int, default=0, help='*select the GPU ID to use')
    parser.add_argument('--use_multi_gpu', default=False, help='*decide whether to use multiple GPUs or not')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='*determine the directory for checkpoints')
    parser.add_argument('--pred_len', type=int, default=720, help='*determine the prediction length')
    parser.add_argument('--label_len', type=int, default=48, help='*determine the label length')
    parser.add_argument('--seq_len', type=int, default=96, help='*determine the length of input sequence')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--embed', type=str, default='timeF', help='*determine the embedding type')
    parser.add_argument('--freq', type=str, default='h', help='*decide frequency of time series data (e.g., h for hourly, d for daily)')
    parser.add_argument('--root_path', type=str, default='./', help='*determine the root directory of dataset')
    parser.add_argument('--data', type=str, default='custom', help='*specify the type of data to use (custom or pred)')
    parser.add_argument('--data_path', type=str, default='datasets/exchange_rate/exchange_rate.csv')
    parser.add_argument('--features', type=str, default='M', choices=['M', 'S'], help='*determine the type of features (M for multivariate, S for single)')
    parser.add_argument('--target', type=str, default='OT', help='determine the target column to predict')
    parser.add_argument('--num_workers', type=int, default=10, help='*determine the number of workers')
    parser.add_argument('--patience', type=int, default=3, help='*determine the target column to predict')
    parser.add_argument('--output_attention', action="store_true", help='*determine whether to output attention')
    parser.add_argument('--lradj', type=str, default='type1', help='*determine the scheme of adjusting learning rate')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--exp_name', type=str, default='Experiment', help='*determine the name of this experiment')
    parser.add_argument('--study_exp', type=str, default='weather_96_96', help='*differ different study file')
    parser.add_argument('--optuna', default=False, help='*determine whether to use optuna or existing checkpoint')
    parser.add_argument('--optuna_direction', type=str, default="minimize", help='*determine the direction of optuna optimization')
    parser.add_argument('--criterion', default=['MAE','MSE'],
                        help='*select the loss function, but the last function is what you want to backward')
    parser.add_argument('--trials', type=int, default=20, help='*determine the number of trials')
    parser.add_argument('--use_norm', type=int, default=True)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_state', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--activation', default='gelu')
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--enc_in', type=int, default=8)
    parser.add_argument('--dec_in', type=int, default=8)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--c_out', type=int, default=8)
    parser.add_argument('--class_strategy', type=str, default='projection')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--distil', action='store_false',default=True)
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--efficient_training', type=bool, default=False)
    parser.add_argument('--partial_start_index', type=int, default=0)
    parser.add_argument('--period', type=int, default=24)
    parser.add_argument('--trend_kernel', type=int, default=25)
    parser.add_argument('--router_num', type=int, default=4)
    parser.add_argument('--STL', type=bool, default=True)

    return parser.parse_args()

args = parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

def main():

    folder = './output/exp_log'
    folder2 = args.model
    folder3 = args.model_id
    subfolder = os.path.join(folder, folder2)
    os.makedirs(subfolder, exist_ok=True)
    subsubfolder = os.path.join(subfolder, folder3)
    log_file = open(subsubfolder, 'w', encoding='utf-8', buffering=1)

    print('Args in experiment:')
    print(args)

    exp = ExpForecast(args)

    exp.train(args.exp_name)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(args.exp_name))
        exp.predict(args.exp_name, True)

    torch.cuda.empty_cache()

    log_file.close()


if __name__ == '__main__':
    main()
    print("-----------Finished-----------")
