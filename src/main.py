import os
import sys
from typing import List
from argparse import ArgumentParser

sys.path.append('./')
from experiment_cifar10 import ExperimentCIFAR10


def str2list(s: str) -> List[str]:
    s = s.lstrip('[').rstrip(']')
    s = s.replace(' ', '')
    s = s.split(',')
    return s


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='CIFAR10')
    parser.add_argument('--dir_name', type=str, default='result')
    parser.add_argument('--scheduler_name', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='ResNet20')

    args = parser.parse_args()
    experiment_name = args.experiment_name
    dir_name = args.dir_name
    scheduler_name = args.scheduler_name
    model_name = args.model_name

    args_dic = dict(model_name=model_name,
                        dir_name=dir_name,
                        max_epoch=200,
                        batch_size=2**2,
                        optimizer_names=None,
                        learning_rate=1e-3,
                        m_list=None,
                        a_list=None,
                        scheduler_name=scheduler_name)

    experiment = ExperimentCIFAR10(**args_dic)
    experiment.experiment()

