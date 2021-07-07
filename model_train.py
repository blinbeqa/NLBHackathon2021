

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



def main(args):


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


    train_data_path = "/home/blin/PycharmProjects/nlb/train_split_train1.npy"

    train_data = Dataset({"path": train_data_path})

    #t_size = int(len(train_data) * 0.66)
    #v_size = len(train_data) - t_size

    #torch.manual_seed(0)
    #train_set, val_set = torch.utils.data.random_split(train_data, [t_size, v_size])

    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=5 * max(num_gpus, 1), drop_last=True)



    train_size = len(train_data_loader) * args.batch_size

    print("train set: ", train_size)

    valid_data_path = "/home/blin/PycharmProjects/nlb/train_split_valid1.npy"

    valid_data = Dataset({"path": valid_data_path})


    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=5 * max(num_gpus, 1), drop_last=True)

    valid_size = len(valid_data_loader) * args.batch_size

    print("valid set: ", valid_size)

    test_data_path = "/home/blin/PycharmProjects/nlb/test1.npy"

    test_data = Dataset({"path": test_data_path})

    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                                    shuffle=False, num_workers=5 * max(num_gpus, 1), drop_last=False)

    test_size = len(test_data_loader)

    print("test set: ", test_size)

    ## Main script for training and validation

    # loss function for training


    loss_func1 = lambda pred, origin: (
            F.binary_cross_entropy_with_logits(pred, origin, reduction="mean"))

    loss_func2 = lambda pred, origin: (

            F.mse_loss(pred, origin, reduction="mean"))

    focal_l = FocalLoss()

    loss_func = lambda pred, origin: (
            F.l1_loss(pred, origin, reduction="mean"))

    # scheduling sampling
    ssr_decay_mode = False
    scheduled_sampling_ratio = 1

    # optimizer and learning rate
    lr_decay_mode = False


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.LBFGS(model.parameters(), lr=args.learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)

    # best model in validation loss
    min_epoch, min_loss = 0, float("inf")

    best_value = 0.
    f1score_best = 0.

    for epoch in range(0, args.num_epochs):

        ## Phase 1: Learning on the training set
        model.train()

        samples = 0
        count = 0
        LOSS_train = 0
        for frames in train_data_loader:
            samples += args.batch_size
            count += 1

            frames = frames.to(device).float()

            inputs = frames[:, :-1]
            #print(inputs.size())

            origin = frames[:, -1:]

            #print(origin.size())


            pred = model(inputs)
            #print("pred: ", pred)

            loss = loss_func2(pred, origin)
            #loss = focal_l(pred, origin)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            #pred = F.sigmoid(pred)

            #pred = torch.round(pred)

        #     LOSS_train += loss_func(pred, origin).item() * args.batch_size
        #
        # LOSS_train /= train_size
        # print("Epoch : ", epoch)
        # print("Train Loss: ", LOSS_train)
        # print("Accuracy Train: ", (1.0 - LOSS_train) * 100.0)



            #print('Epoch: {}/{}, Training: {}/{}, Loss: {}'.format(epoch + 1, args.num_epochs, samples, train_size, loss.item()))

        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay_rate



            ## Phase 2: Evaluation on the validation set
        model.eval()

        gt_values = []
        pred_values = []



        with torch.no_grad():

            samples, LOSS, LOSS_v = 0, 0., 0.
            for frames in valid_data_loader:
                samples += args.batch_size

                frames = frames.to(device).float()


                inputs = frames[:, :-1]
                #print(inputs)

                origin = frames[:, -1:]


                pred = model(inputs)
                #pred = F.sigmoid(pred)

                pred = torch.round(pred)

                #print("pred: ", pred)
                #print("orig: ", origin)
                #print("l            ", loss_func2(pred, origin).item())
                #print("lb ",loss_func2(pred, origin).item() * args.batch_size)


                LOSS += loss_func(pred, origin).item() * args.batch_size


        #         for i in range(len(pred)):
        #             gt_values.append(origin[i].cpu().numpy())
        #             pred_values.append(pred[i].cpu().numpy())
        #
        #         #LOSS_v += loss_func(pred, origin).item()
        #         #LOSS += focal_l(pred, origin).item() * args.batch_size
        #
        # gt_values = numpy.array(gt_values)
        # pred_values = numpy.array(pred_values)
        # # #print(gt_values)
        # accuracy = accuracy_score(gt_values, pred_values)
        # f1score = f1_score(gt_values, pred_values)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))
        # print("F1 :", f1score)

        LOSS /= valid_size
        #LOSS_v /= len(valid_data_loader)
        print("Epoch : ", epoch)
        print("Valid Loss: ", LOSS)
        print("Accuracy: ", (1.0 - LOSS) * 100.0)
        # print("Valid Loss2: ", LOSS_v)
        # print("Accuracy2: ", (1.0 - LOSS_v) * 100.0)
        # if LOSS < min_loss:
        #     min_epoch, min_loss = epoch + 1, LOSS

        if ((1.0 - LOSS) * 100.0) > best_value:

            best_value = ((1.0 - LOSS) * 100.0)
            #f1score_best = f1score
            torch.save(model.state_dict(), "model_best.pt")
            print("Saved                      Saved")

    print("Best value Accuracy ", best_value)
    #print("Best value F1: ", f1score_best)

    #SHAP example for demonstartion purposes only

    batch = next(iter(valid_data_loader))
    print(batch)
    # images, _ = batch
    batch = batch.to(device).float()

    background = batch[1:24, :-1]
    test = batch[0:1, :-1]

    e = shap.DeepExplainer(model, background)
    shap_values = e.shap_values(test)

    print("shap values", shap_values)
    print("Evaluation", batch[0:1, 15])

    shap.summary_plot(shap_values, feature_names=["ApplicantGender", "ApplicantMarried", "ApplicantDependents", "ApplicantEducation", "ApplicantSelfEmployed", "ApplicantIncome", "ApplicantCreditHistory", "ApplicantZIP", "ApplicantState", "ApplicantEmplLength", "ApplicantHomeOwn", "LoanAmount", "LoanTerm", "LoanIntRate", "LoanPurpose"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLB")



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




    parser.add_argument('--use-sigmoid', dest='use_sigmoid', action='store_true',
                        help='Use sigmoid function at the output of the model.')
    parser.add_argument('--no-sigmoid', dest='use_sigmoid', action='store_false',
                        help='Use output from the last layer as the final output.')
    parser.set_defaults(use_sigmoid=False)


    ## Dataset (Input to the training algorithm)
    parser.add_argument('--dataset', default="NLB", type=str,
                        help='The dataset name. (Options: MNIST, KTH)')



    ## Learning algorithm
    parser.add_argument('--num-epochs', default=99, type=int,
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
    parser.add_argument('--lr-decay-rate', default=0.95, type=float,
                        help='The learning rate by decayed by decay_rate every lr_decay_epoch.')

    # scheduled sampling ratio
    parser.add_argument('--ssr-decay-epoch', default=1, type=int,
                        help='Decay the scheduled sampling every ssr_decay_epoch.')
    parser.add_argument('--ssr-decay-ratio', default=4e-3, type=float,
                        help='Decay the scheduled sampling by ssr_decay_ratio every time.')

    main(parser.parse_args())
