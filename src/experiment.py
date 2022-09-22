import os
import sys
import math
import pickle
import random
import numpy as np
import torch
from typing import Any, Dict, Iterable, List
from abc import ABC, abstractmethod
from torch.cuda import is_available as torch_cuda_is_available
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from torch.optim import SGD
from torch.optim import Adam

sys.path.append('./')

scheduler_dic = dict(div_sqrt=lambda optimizer: LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (math.sqrt(epoch) + 1)))


class Experiment(ABC):
    """
    実験の抽象class

    Attributes
    ----------
    model_dic: dict
        学習させるmodel の辞書．
    dataset_name: str
        Dataset の名前．
    max_epoch: int
        学習させるepoch 数の最大値．
    record_cols: str
        記録する評価指標名(column) 名．
    batch_size: int
        Batch size．
    optimizer_names: list of str
        実験対象の最適化algorithm．
    learning_rate: float
        学習率 (Step size)．
    m_list: list of str
        Grid Search の際に検討するm の値のlist．
    a_list: list of str
        Grid Search の際に検討するa の値のlist．
    root: str
        この直下にDirectory を掘って結果を保存．
    device: str
        学習時にCPU とGPU のどちらを用いるか．

    optimizer_dic: dict
        最適化algorithm の辞書．
    exp_dic:

    """
    def __init__(self, model_dic: Dict[str, Any], dataset_name: str, model_name: str, dir_name: str,
                 max_epoch: int, record_cols: List[str], batch_size: int, optimizer_names: List[str],
                 learning_rate: float, m_list: List[str] = None, a_list: List[str] = None, root: str = '../',
                 device: str = None, scheduler_name: str = None):

        self.model_dic = model_dic
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.record_cols = record_cols

        self.optimizer_dic = {'Momentum': SGD , 'Adam': Adam}

        self.optimizer_names = optimizer_names if optimizer_names else list(self.optimizer_dic)
        if type(self.optimizer_names) is str:
            self.optimizer_names = [self.optimizer_names]

        if device:
            self.device = device
        elif torch_cuda_is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        print('Using {} ...'.format(self.device))

        # Learning Rate
        self.scheduler_name = scheduler_name

        # Path Setting
        self.root = root
        self.data_dir = os.path.join(self.root, 'data')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        self.result_dir = os.path.join(self.root)
        for d in ['result', dataset_name, model_name, dir_name]:
            self.result_dir = os.path.join(self.result_dir, d)
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)

        if self.scheduler_name:
            self.result_dir = os.path.join(self.result_dir, self.scheduler_name)
            if not os.path.isdir(self.result_dir):
                os.mkdir(self.result_dir)

        self.train_loader = None
        self.test_loader = None
        self.net = None

    @abstractmethod
    def _choice_model(self, name: str) -> torch.nn.Module:
        raise NotImplementedError

    def choice_optimizer(self, name: str, params, **kwargs):
        if name in self.optimizer_dic:
            kwargs['lr'] = 1e-3
            if 'SGD' in name:
                pass
            if 'Momentum' in name:
                kwargs['momentum'] = 0.9
            if 'Adam' in name:
                kwargs['amsgrad'] = False
                kwargs['betas'] = (0.9, 0.999)
            print(kwargs)
            return self.optimizer_dic[name](params, **kwargs)
        else:
            raise ValueError('Invalid optimizer name was given ...')

    def save_model(self, obj: object, pkl_name: str, dir_name=None) -> None:
        result_dir = self.result_dir
        if dir_name:
            result_dir = os.path.join(result_dir, dir_name)
        with open(os.path.join(result_dir, pkl_name), 'wb') as f:
            pickle.dump(obj, f)

    def to_csv(self, dic: Dict[str, List[float]], csv_name: str, dir_name: str = None) -> None:
        result_dir = self.result_dir
        if dir_name:
            result_dir = os.path.join(result_dir, dir_name)

        def to_line(l: Iterable) -> str:
            l = list(map(str, l))
            return ','.join(l) + '\n'

        with open(os.path.join(result_dir, csv_name), 'w') as f:
            keys = list(dic)
            f.write(to_line(keys))
            for value in zip(dic.values()):
                line = to_line(value)
                f.write(line)

    @abstractmethod
    def _prepare_data(self):
        raise NotImplementedError

    @abstractmethod
    def _batch_train(self, net: torch.nn.Module, train_loader: DataLoader, optimizer):
        raise NotImplementedError

    @abstractmethod
    def _validate(self, net: torch.nn.Module, test_loader: DataLoader) -> List[float]:
        raise NotImplementedError

    def train(self, net: torch.nn.Module, optimizer, use_tqdm=True) -> List[List[float]]:
        epochs = range(self.max_epoch)
        if use_tqdm:
            epochs = tqdm(epochs)

        if self.scheduler_name:
            scheduler = scheduler_dic[self.scheduler_name](optimizer)
        else:
            scheduler = None

        records = [np.nan for _ in range(self.max_epoch)]
        for epoch in epochs:
            net, train_score = self._batch_train(net, self.train_loader, optimizer)
            test_score = self._validate(net, self.test_loader)
            records[epoch] = [*train_score, *test_score]
            if scheduler:
                scheduler.step()

        return records

    def save_result(self, records: List[List[float]], csv_path: str):
        def to_line(l: Iterable) -> str:
            l = list(map(str, l))
            return ','.join(l) + '\n'

        with open(csv_path, 'w') as f:
            keys = self.record_cols
            f.write(to_line(keys))
            for record in records:
                line = to_line(record)
                f.write(line)

    def experiment(self) -> None:
        self.train_loader, self.test_loader = self._prepare_data()
        for optimizer_name in self.optimizer_names:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

            print(optimizer_name)
            csv_name = optimizer_name[:]
            csv_name = '{}.csv'.format(csv_name)
            csv_path = os.path.join(self.result_dir, csv_name)
            if not os.path.isfile(csv_path):
                net = self._choice_model(self.model_name).to(self.device)
                optimizer = self.choice_optimizer(optimizer_name, net.parameters())
                records = self.train(net, optimizer)
                self.save_result(records, csv_path)
            else:
                print('already exists')
