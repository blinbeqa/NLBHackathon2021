# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

# system modules
import os, argparse

# basic pytorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy


# custom utilities
from dataloader import NLB_Dataset
from model import MLP
from focal_loss import FocalLoss

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import f1_score


def main(args):
    # train_data_path = "/home/blin/PycharmProjects/nlb/train_split2.npy"
    # data = numpy.load(train_data_path)
    # data_features = data[:, :-1]
    # data_res = data[:, -1:]
    #
    # # split data into train and test sets
    # seed = 7
    # test_size = 0.33
    # X_train, X_test, y_train, y_test = train_test_split(data_features, data_res, test_size=test_size, random_state=seed)

    train_data_path = "/home/blin/PycharmProjects/nlb/train_split_train1.npy"
    valid_data_path = "/home/blin/PycharmProjects/nlb/train_split_valid1.npy"

    data_t = numpy.load(train_data_path)
    X_train = data_t[:, :-1]
    y_train = data_t[:, -1:]

    data_v = numpy.load(valid_data_path)
    X_test = data_v[:, :-1]
    y_test = data_v[:, -1:]

    # tl = TomekLinks()
    #
    # X_train, y_train = tl.fit_resample(X_train, y_train)
    # X_test, y_test = tl.fit_resample(X_test, y_test)

    # fit model no training data
    model = XGBClassifier()
    # model = XGBClassifier(silent=False,
    #                       scale_pos_weight=1,
    #                       learning_rate=0.1,
    #                       colsample_bytree=0.4,
    #                       subsample=0.8,
    #                       objective='binary:logistic',
    #                       n_estimators=6000,
    #                       reg_alpha=0.3,
    #                       max_depth=5,
    #                       gamma=10)
    #model = RandomForestClassifier()
    model.fit(X_train, y_train)



    print(model)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]

    accuracy = accuracy_score(y_test, predictions)
    f1score = f1_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("F1 :", f1score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conv-TT-LSTM Training")

    ## Data format (batch_size x time_steps x height x width x channels)

    # batch size and the logging period
    parser.add_argument('--batch-size', default=16, type=int,
                        help='The batch size in training phase.')


    ## Devices (CPU, single-GPU or multi-GPU)

    # whether to use GPU for training
    parser.add_argument('--use-cuda', dest='use_cuda', action='store_true',
                        help='Use GPU for training.')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false',
                        help="Do not use GPU for training.")
    parser.set_defaults(use_cuda=True)

    # whether to use multi-GPU for training
    parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                        help='Use multiple GPUs for training.')
    parser.add_argument('--single-gpu', dest='multi_gpu', action='store_false',
                        help='Do not use multiple GPU for training.')
    parser.set_defaults(multi_gpu=True)

    ## Models (Conv-LSTM or Conv-TT-LSTM)


    parser.add_argument('--use-sigmoid', dest='use_sigmoid', action='store_true',
                        help='Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid', dest='use_sigmoid', action='store_false',
                        help='Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid=False)


    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default="NLB", type=str,
                        help='The dataset name. (Options: MNIST, KTH)')

    # training set
    parser.add_argument('--train-data-file', default='moving-mnist-train.npz', type=str,
                        help='Name of the folder/file for training set.')
    parser.add_argument('--train-samples', default=10000, type=int,
                        help='Number of unique samples in training set.')

    # validation set
    parser.add_argument('--valid-data-file', default='moving-mnist-val.npz', type=str,
                        help='Name of the folder/file for validation set.')
    parser.add_argument('--valid-samples', default=3000, type=int,
                        help='Number of unique samples in validation set.')

    ## Learning algorithm
    parser.add_argument('--num-epochs', default=400, type=int,
                        help='Number of total epochs in training.')

    parser.add_argument('--decay-log-epochs', default=20, type=int,
                        help='The window size to determine automatic scheduling.')

    # gradient clipping
    parser.add_argument('--gradient-clipping', dest='gradient_clipping',
                        action='store_true', help='Use gradient clipping in training.')
    parser.add_argument('--no-clipping', dest='gradient_clipping',
                        action='store_false', help='No gradient clipping in training.')
    parser.set_defaults(use_clipping=False)

    parser.add_argument('--clipping-threshold', default=3, type=float,
                        help='The threshold value for gradient clipping.')

    # learning rate scheduling
    parser.add_argument('--learning-rate', default=0.001, type=float,
                        help='Initial learning rate of the Adam optimizer.')
    parser.add_argument('--lr-decay-epoch', default=5, type=int,
                        help='The learning rate is decayed every lr_decay_epoch.')
    parser.add_argument('--lr-decay-rate', default=0.90, type=float,
                        help='The learning rate by decayed by decay_rate every lr_decay_epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr-decay-epoch', default=1, type=int,
                        help='Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default=4e-3, type=float,
                        help='Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
