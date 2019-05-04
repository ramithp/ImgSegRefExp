import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models
import numpy as np

def init_weights(m):
    if type(m) == nn.Conv2d :
        print("init conv")
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        print("init bn")
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.Linear:
        print("Init linear")
        torch.nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.GRU or type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if('weight' in name):
                print("initializing LSTM/GRU weight ", name)
                torch.nn.init.xavier_normal_(param)
            elif 'bias' in name:
                print("bias init", name)
                torch.nn.init.constant_(param, 0.0)


def conv_relu(kernel_size, stride, in_channels, out_channels, padding=0, bias=True):
    #TODO: weight init
    layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                      nn.ReLU(inplace=True))
    return layer

def conv(kernel_size, stride, in_channels, out_channels, padding=0, bias=True):
    #TODO: weight init
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    return layer
    
def generate_spatial_batch(N, featmap_H, featmap_W):
    spatial_batch_val = np.zeros((N, featmap_H, featmap_W, 8), dtype=np.float32)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w+1) / featmap_W * 2 - 1
            xctr = (xmin+xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h+1) / featmap_H * 2 - 1
            yctr = (ymin+ymax) / 2
            spatial_batch_val[:, h, w, :] = \
            [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
            
    return torch.Tensor(spatial_batch_val).to(device)