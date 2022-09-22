import os
import sys
import torch
from time import time
from tqdm import tqdm
from typing import List

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
sys.path.append('./')
from experiment import Experiment
from model.resnet import resnet20, resnet32, resnet44, resnet56, resnet110

model_dic = dict(ResNet18=resnet18, ResNet20=resnet20, ResNet32=resnet32, ResNet44=resnet44, ResNet56=resnet56, ResNet110=resnet110)


class ExperimentCIFAR10(Experiment):
    def __init__(self, model_name: str, dir_name: str, max_epoch: int, batch_size: int,
                 optimizer_names: List[str], learning_rate: float, m_list: List[str] = None, a_list: List[str] = None,
                 root:  str = '../', device: str = None, scheduler_name: str = None):
        dataset_name = 'CIFAR10'
        record_cols = ['loss', 'train_acc', 'time', 'test_acc']
        super(ExperimentCIFAR10, self).__init__(model_dic, dataset_name, model_name, dir_name, max_epoch, record_cols,
                                                batch_size, optimizer_names, learning_rate, m_list, a_list, root,
                                                device, scheduler_name)

    def _prepare_data(self):
        root = os.path.join(self.data_dir, self.dataset_name)
        train_data = CIFAR10(root, train=True, download=True, transform=ToTensor())
        test_data = CIFAR10(root, train=False, download=True, transform=ToTensor())
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def _choice_model(self, name: str) -> torch.nn.Module:
        return self.model_dic[name]()

    def _batch_train(self, net: torch.nn.Module, train_loader: DataLoader, optimizer):
        running_loss = 0.0
        i = 0
        total = 0
        correct = 0
        criterion = CrossEntropyLoss()
        start = time()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            i += 1

        return net, [running_loss / i, correct / total, time() - start]

    def _validate(self, net: torch.nn.Module, test_loader: DataLoader) -> List[float]:
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return [correct / total]

