

import torch
import torch.nn as nn


## Multi layer perceptron
class MLP(nn.Module):
    def __init__(self,
                 # input to the model
                 input_channels,
                 # architecture of the model
                 hidden_channels,

                 # output function and output format
                 output_sigmoid=False):

        super(MLP, self).__init__()

        ## Hyperparameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.layers = nn.ModuleDict()

        self.layers["input"] = torch.nn.Linear(self.input_channels, self.hidden_channels[0])

        torch.nn.init.xavier_uniform(self.layers["input"].weight)
        self.layers["norminput"] = torch.nn.BatchNorm1d(self.hidden_channels[0])
        self.layers["reluinput"] = torch.nn.LeakyReLU(0.001)
        #self.layers["reluinput"] = torch.nn.ReLU()
        #self.layers["reluinput"] = torch.nn.Sigmoid()
        #self.layers["dropinput"] = nn.Dropout(0.3)

        for l in range(len(self.hidden_channels) - 1):

            channels_i = hidden_channels[l]
            channels_o = hidden_channels[l+1]

            lid = "l{}".format(l)  # layer ID
            lid_relu = "relu{}".format(l)  # layer ID
            lid_norm = "norm{}".format(l)  # layer ID
            lid_drop = "drop{}".format(l)  # layer ID
            self.layers[lid] = torch.nn.Linear(channels_i, channels_o)
            torch.nn.init.xavier_uniform(self.layers[lid].weight)

            self.layers[lid_norm] = torch.nn.BatchNorm1d(channels_o)
            self.layers[lid_relu] = torch.nn.LeakyReLU(0.001)
            #self.layers[lid_relu] = torch.nn.ReLU()
            #self.layers[lid_relu] = torch.nn.Sigmoid()
            #self.layers[lid_drop] = nn.Dropout(0.1)


        self.layers["output"] = torch.nn.Linear(self.hidden_channels[-1], 1)
        torch.nn.init.xavier_uniform(self.layers["output"].weight)


    def forward(self, input):



        input = self.layers["input"](input)
        input = self.layers["norminput"](input)
        #identity = input
        input = self.layers["reluinput"](input)
        #input = self.layers["dropinput"](input)


        for l in range(len(self.hidden_channels) - 1):
            lid = "l{}".format(l)  # layer ID
            lid_relu = "relu{}".format(l)  # layer ID
            lid_norm = "norm{}".format(l)  # layer ID
            lid_drop = "drop{}".format(l)  # layer ID
            input = self.layers[lid](input)
            input = self.layers[lid_norm](input)
            #input += identity
            input = self.layers[lid_relu](input)
            #input = self.layers[lid_drop](input)

        output = self.layers["output"](input)

        return output