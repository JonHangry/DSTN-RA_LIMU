from data_provider.data_factory import data_provider
from exp_basic import ExpBasic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import scipy as sp
from scipy.stats import pearsonr,spearmanr
import pandas as pd
import random
from scipy.special import logsumexp
from scipy.stats import norm

# Please install these libs

warnings.filterwarnings('ignore')

class ExpForecast(ExpBasic):

    def __init__(self, args):
        super(ExpForecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):

        if self.args.optimizer == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'AdamW':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            return None
            # model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def Likelihood_Gaussian(self,outputs,batch_y,var):
        residuals = batch_y - outputs
        residuals_tensor = torch.tensor(residuals, dtype=torch.float32)
        var = torch.mean(residuals_tensor ** 2)
        len_pred = batch_y.shape[1]
        log_prob = -0.5 * len_pred - 0.5 * len_pred * torch.log(2 * torch.pi * var)
        log_prob = torch.mean(log_prob)
        return log_prob

    def Likelihood_mixture(self, outputs, batch_y, logvar):
        num = torch.tensor(batch_y.shape[0])
        len_pred = torch.tensor(batch_y.shape[1])
        outputs = outputs.transpose(1,0)
        batch_y = batch_y.transpose(1,0)
        likelihood = -0.5 * len_pred * (torch.log(torch.tensor(2.0) * torch.pi) + logvar)
        residuals = torch.square(outputs - batch_y)
        s_residuals = torch.sum(residuals, 0)
        s_residuals = s_residuals.reshape((num, 1, 1))
        likelihood += -0.5 * s_residuals * torch.exp(-logvar)
        likelihood = torch.logsumexp(likelihood, dim=0)
        return torch.mean(likelihood)

    def Calibration(self, outputs, batch_y,var):
        N,L = batch_y.shape
        logvar = var.squeeze(dim=2).detach().cpu().numpy()
        std = np.sqrt(np.exp(logvar))
        std = np.broadcast_to(std,(N,L))
        samples = np.random.normal(loc=outputs.detach().cpu().numpy()[..., None],
                                   scale=std[..., None],
                                   size=(outputs.shape[0],
                                         outputs.shape[1],
                                         30))
        # samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
        cal_error = np.zeros(outputs.shape[1])
        cal_error2 = np.zeros(outputs.shape[1])
        cal_thresholds = np.linspace(start=0, stop=1, num=11)
        for p in cal_thresholds:
            q = np.quantile(samples, p, axis=-1)
            # q = np.expand_dims(q, axis=-1)
            est_p = np.mean(batch_y.detach().cpu().numpy() <= q, axis=0)
            cal_error += np.mean((est_p - p) ** 2, axis=-1)
            # cal_error2 = (est_p - p) ** 2
            cal_error_scalar = np.mean(cal_error)
        # print("cal_error:{}".format(cal_error2))
        return cal_error_scalar


    def IC(self, x, y):
        return np.mean(x)

    def RankIC(self, x, y):
        return np.mean(y)

    def ICIR(self, x,y):
        return np.mean(x)/np.std(x)

    def RankICIR(self, x,y):
        return np.mean(y)/np.std(y)

    def _select_criterion(self, n):
        if self.args.criterion[n] == 'CrossEntropy':
            criterion = nn.CrossEntropyLoss()
        elif self.args.criterion[n] == 'MSE':
            criterion = nn.MSELoss()
        elif self.args.criterion[n] == 'MAE':
            criterion = nn.L1Loss()
        elif self.args.criterion[n] == 'Likelihood_Gaussian':
            criterion = self.Likelihood_Gaussian
        elif self.args.criterion[n] == 'Likelihood_mixture':
            criterion = self.Likelihood_mixture
        elif self.args.criterion[n] == 'Calibration':
            criterion = self.Cal_Gluformer
        elif self.args.criterion[n] == 'IC':
            criterion = self.IC
        elif self.args.criterion[n] == 'RankIC':
            criterion = self.RankIC
        elif self.args.criterion[n] == 'ICIR':
            criterion = self.ICIR
        elif self.args.criterion[n] == 'RankICIR':
            criterion = self.RankICIR
        else:
            print("You give a wrong criterion")
            criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        test_loss, vali_loss = None, None
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=100)

        criterion_len = len(self.args.criterion)
        loss_dict = {key: [] for key in self.args.criterion}
        scaler = None

        if self.args.use_amp:
            scaler = torch.amp.GradScaler("cuda")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        for num_loss in range(criterion_len):
                            # print(num_loss)
                            criterion = self._select_criterion(num_loss)
                            loss = criterion(outputs, batch_y)
                            if self.args.criterion[num_loss] in loss_dict:
                                loss_dict[self.args.criterion[num_loss]].append(loss.item())

                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = nn.MSELoss()(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader)
            test_loss = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} |MSE Train Loss: {2:4f} Vali Loss: {3:4f} Test Loss: {4:4f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        vali_loss = self.vali_test(vali_data, vali_loader)
        print("vali_loss: {}".format(vali_loss))
        test_loss = self.vali_test(test_data, test_loader)
        print("test_loss: {}".format(test_loss))

        return self.model, test_loss

    def vali(self, vali_data, vali_loader):
        total_loss = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = nn.MSELoss()(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali_test(self, vali_data, vali_loader):
        total_loss = []
        ic_list = []
        rank_ic_list = []

        self.model.eval()
        criterion_len = len(self.args.criterion)
        loss_dict = {key: [] for key in self.args.criterion}
        scaler = None

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                if (i % 100 == 0):
                    print("its batch:{}",format(i))
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                B, L, N = outputs.shape
                pred = outputs.transpose(2, 1).reshape(B * N, L)
                true = batch_y.transpose(2, 1).reshape(B * N, L)
                if any(x in self.args.criterion for x in ['IC', 'RankIC','IR','RankIR']):               
                    for b in range(true.shape[1]):
                        pred_indices = torch.argsort(pred, dim=0)
                        true_indices = torch.argsort(true, dim=0)
                        pred_ranks = torch.zeros_like(pred_indices, dtype=torch.float32)
                        true_ranks = torch.zeros_like(true_indices, dtype=torch.float32)
                        device = pred_ranks.device
                        for col in range(pred.shape[1]):
                            pred_ranks[pred_indices[:, col], col] = torch.arange(pred.shape[0], dtype=torch.float32, device=device)
                            true_ranks[true_indices[:, col], col] = torch.arange(true.shape[0], dtype=torch.float32, device=device)
                        cov = torch.mean((pred - pred.mean(dim=0)) * (true - true.mean(dim=0)), dim=0)
                        cov_rank = torch.mean((pred_ranks - pred_ranks.mean(dim=0)) * (true_ranks - true_ranks.mean(dim=0)),
                                              dim=0)
                        pred_std = pred.std(dim=0)
                        pred_rank_std = pred_ranks.std(dim=0)
                        true_std = true.std(dim=0)
                        true_rank_std = true_ranks.std(dim=0)
                        ic = cov / (pred_std * true_std + 1e-8)
                        rank_ic = cov_rank / (pred_rank_std * true_rank_std + 1e-8)
                        ic = ic.mean()
                        rank_ic = rank_ic.mean()
                        ic_list.append(ic.detach().cpu().numpy())
                        rank_ic_list.append(rank_ic.detach().cpu().numpy())
                for num_loss in range(criterion_len):
                    # print(num_loss)
                    criterion = self._select_criterion(num_loss)
                    if self.args.criterion[num_loss] not in ['IC','RankIC','ICIR','RankICIR']:
                        if self.args.criterion[num_loss] in ['Likelihood_Gaussian','Likelihood_mixture','Calibration']:
                            loss = criterion(pred, true)
                        else:
                            loss = criterion(outputs, batch_y)
                        if self.args.criterion[num_loss] in loss_dict:
                            loss_dict[self.args.criterion[num_loss]].append(loss.item())

            for num_loss in range(criterion_len):
                criterion = self._select_criterion(num_loss)
                if self.args.criterion[num_loss] in ['IC', 'RankIC', 'ICIR', 'RankICIR']:
                    loss = criterion(ic_list, rank_ic_list)
                    if self.args.criterion[num_loss] in loss_dict:
                        loss_dict[self.args.criterion[num_loss]].append(loss.item())

        avg_vali_loss = {k: np.mean(v) for k, v in loss_dict.items()}
        self.model.train()
        return avg_vali_loss

    def get_input(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        inputs = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            input = batch_x.detach().cpu().numpy()
            inputs.append((input))
        folder_path = './results/' + setting + '/'
        np.save(folder_path + 'input.npy', inputs)

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.amp.autocast("cuda"):
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return