import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models

import utils
import argparse

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--print_every', dest="print_every", action="store", type=int, default=5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
where = pa.data_dir
path = pa.save_dir
learning_rate = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs
print_every = pa.print_every

# print(structure)
trainloader, testloader, validloader, train_data, test_data, validation_data, train_dir ,test_dir, valid_dir = utils.load_data(where)
# print(train_data)

model, criterion, optimizer = utils.create_model(structure,dropout, learning_rate, output_labels=102)


utils.train_model(model, trainloader, validloader, criterion, optimizer, epochs, print_every, power)
utils.check_accuracy(model, testloader)

utils.save_checkpoint(model, train_data, epochs, optimizer, path)

print("----------------------------------------")
print("The model has been created, trained and saved")