
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
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

import shap
import csv



def main(args):
    ## Model preparation (Conv-LSTM or Conv-TT-LSTM)

    # whether to use GPU (or CPU)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # whether to use multi-GPU (or single-GPU)
    multi_gpu = use_cuda and args.multi_gpu and torch.cuda.device_count() > 1
    num_gpus = (torch.cuda.device_count() if multi_gpu else 1) if use_cuda else 0

    # construct the model with the specified hyper-parameters
    model = MLP(
        # input to the model
        input_channels=15,
        # architecture of the model
        hidden_channels=(100, 50),
        output_sigmoid=args.use_sigmoid)

    # move the model to the device (CPU, GPU, multi-GPU)
    model.to(device)
    if multi_gpu:
        model = nn.DataParallel(model)

    ## Dataset Preparation (Moving-MNIST, KTH)
    Dataset = {"NLB": NLB_Dataset}[args.dataset]



    test_data_path = "/home/blin/PycharmProjects/nlb/test1.npy"

    test_data = Dataset({"path": test_data_path})

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                    shuffle=False, num_workers=5 * max(num_gpus, 1), drop_last=False)

    test_size = len(test_data_loader)

    print("test set: ", test_size)



    model.load_state_dict(torch.load("model_best.pt"))


    model.eval()

    all_lines = []
    with torch.no_grad():

        for frames in test_data_loader:

            frames = frames.to(device).float()

            inputs = frames[:, 2:]

            ids = frames[:, 0:2]

            pred = model(inputs)

            pred = torch.round(pred)

            pred = pred[0].cpu().numpy()
            ids = ids[0].cpu().numpy()

            line = ["{number:06}".format(number=int(ids[1])), "{number:06}".format(number=int(ids[0])), int(pred[0])]
            all_lines.append(line)
            #print(line)

    # fields = []
    # rows = []
    # filename = "/home/blin/Downloads/test1.csv"
    # with open(filename, 'r') as csvfile:
    #     # creating a csv reader object
    #     csvreader = csv.reader(csvfile)
    #
    #     # extracting field names through first row
    #     fields = next(csvreader)
    #
    #     # extracting each data row one by one
    #     for row in csvreader:
    #         rows.append(row)
    #
    # for i in range(len(rows)):
    #     if rows[i][1] != all_lines[i][1]:
    #         print("errorrrrrrrrrrrrrrrrr")


    with open('result_final.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['ApplicantID', 'LoanID', 'LoanApproved'])
        writer.writerows(all_lines)









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLB")


    parser.add_argument('--batch-size', default=24, type=int,
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




    parser.add_argument('--use-sigmoid', dest='use_sigmoid', action='store_true',
                        help='Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid', dest='use_sigmoid', action='store_false',
                        help='Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid=False)


    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default="NLB", type=str,
                        help='The dataset name. (Options: NLB)')


    ## Learning algorithm
    parser.add_argument('--num-epochs', default=1, type=int,
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
    parser.add_argument('--lr-decay-rate', default=0.97, type=float,
                        help='The learning rate by decayed by decay_rate every lr_decay_epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr-decay-epoch', default=1, type=int,
                        help='Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default=4e-3, type=float,
                        help='Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
