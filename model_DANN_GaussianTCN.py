import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from math import pi
from torch.nn.utils import weight_norm
import torchsummary

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN_FeatureExtractor(nn.Module):
    def __init__(self,feature_layer):
        super(TCN_FeatureExtractor, self).__init__()
        self.cell = TemporalConvNet(num_inputs=5,num_channels=feature_layer)

    def forward(self, x):
        x = x.permute(0,2,1)
        output = self.cell(x)
        output = output.permute(0,2,1)
        return output


class Regressioner(nn.Module):
    def __init__(self):
        super(Regressioner, self).__init__()
        self.regression_mean = nn.Sequential()
        self.regression_mean.add_module("r_mean_fc1",nn.Linear(40,1))
        self.regression_mean.add_module("r_mean_relu1",nn.ReLU(True))
        self.regression_delta = nn.Sequential()
        self.regression_delta.add_module("r_delta_fc1", nn.Linear(40, 1))
        self.regression_delta.add_module("r_delta_relu1", nn.Softplus(True))

    def forward(self, x):
        mean = self.regression_mean(x)
        delta = self.regression_delta(x)
        return mean,delta

    def loss(self,label,mean,delta):
        delta_square = delta * delta
        t1 = 2 * pi * delta_square
        t1 = torch.log(t1)
        t1 = - t1 / 2

        t2 = label - mean
        t2 = torch.pow(t2,2)
        t2 = - t2
        t2 = t2 / (2 * delta_square)

        loss = t1 + t2

        loss = torch.sum(loss)
        loss = -loss

        return loss


class Discriminator(nn.Module):
    def __init__(self, input_dim=40*11, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_fc1', nn.Linear(input_dim, hidden_dim))
        self.discriminator.add_module('d_bn1', nn.BatchNorm1d(hidden_dim))
        self.discriminator.add_module('d_relu1', nn.ReLU(True))
        self.discriminator.add_module('d_fc2', nn.Linear(hidden_dim, 1))
        self.discriminator.add_module('d_sig1', nn.Sigmoid())

    def forward(self, x):
        return self.discriminator(x)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN_GaussianTCN(nn.Module):

    def __init__(self,feature_layer):
        super(DANN_GaussianTCN, self).__init__()
        self.feature = TCN_FeatureExtractor(feature_layer)
        self.regressioner = Regressioner()
        self.domain_discriminator = Discriminator()

    def forward(self,input,alpha=1):
        feature = self.feature(input)
        regression_mean,regression_delta = self.regressioner(feature)
        domain_output = self.get_adversarial_result(feature,alpha)
        return (regression_mean,regression_delta),domain_output

    def get_adversarial_result(self,feature,alpha=1):
        x = ReverseLayerF.apply(feature,alpha)
        x = x.contiguous().view(-1,40*11)
        domain_output = self.domain_discriminator(x)
        return domain_output

    def loss(self,regression_output,domain_output,regression_label,device,source = True):
        regression_loss = self.regressioner.loss(regression_label,regression_output[0],regression_output[1])
        domain_dis_criterion = nn.BCELoss()
        if source:
            domain_label = torch.ones(len(domain_output)).float()
        else:
            domain_label = torch.zeros(len(domain_output)).float()
        domain_output = torch.squeeze(domain_output,dim=-1)
        # print(domain_output)
        domain_label = domain_label.to(device)
        domain_loss = domain_dis_criterion(domain_output,domain_label)
        return regression_loss,domain_loss

if __name__ == "__main__":
    feature_tcn = TCN_FeatureExtractor([40]*4).to("cpu")
    torchsummary.summary(feature_tcn,(11,5),verbose=2)