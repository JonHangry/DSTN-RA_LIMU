# 这里定义一个运行的类
import os
import torch
import warnings

from model import DSTNRA
class ExpBasic(object):
    def __init__(self, args):
        self.args = args

        self.model_dict = {
            "DSTNRA": DSTNRA,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:

            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices

            device = torch.device('cuda:{}'.format(self.args.gpu))
            # 显示使用的GPU
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # 采用CPU
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        pass

    def vali(self, vali_data, vali_loader, criterion):
        pass

    def train(self, setting):
        pass

    def test(self, test_data, test_loader, criterion):
        pass