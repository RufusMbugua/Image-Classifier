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
import json

ap = argparse.ArgumentParser(description='Predict.py')

# Command Line arguments

ap = argparse.ArgumentParser(
    description='predict-file')
ap.add_argument('input_img', default='./flowers/test/21/image_06807.jpg', nargs='*', action="store", type = str)
ap.add_argument('checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
ap.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
ap.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
ap.add_argument('--gpu', default="gpu", action="store", dest="gpu")
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str)


pa = ap.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint
structure = pa.arch
dropout = pa.dropout
learning_rate = pa.learning_rate

# Cat to Name
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
    
# Use the trained model to Predict the Class of the Flower
def predict(image, model, topk=5, device='cuda:0'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    
    image = image.unsqueeze_(0)
    image = image.float()
    image = image.to(device)
    
    class_to_idx_inverted = { val:key for key,val in model.class_to_idx.items() }
    
    with torch.no_grad():
        output = model.forward(image)

    probability = F.softmax(output.data, dim=1)
    
    probabilities,classes = probability.topk(topk)
    
    probabilities = probabilities.cpu().numpy()[0].tolist() 
    list_of_classes = classes.cpu()
    list_of_classes = list_of_classes.data.numpy()[0].tolist()  
    classes = [ class_to_idx_inverted[idx] for idx in list_of_classes ]
    
    # Classes to name
    classes = [ cat_to_name[idx] for idx in classes ]
    
    return (probabilities, classes)

trainloader, testloader, validloader, train_data, test_data, validation_data, train_dir ,test_dir, valid_dir = utils.load_data('./flowers/')

model, criterion, optimizer = utils.create_model(structure,dropout, learning_rate, output_labels=102)
model = utils.load_checkpoint(model, path)

image = (input_img)
image = utils.process_image(image)

probabilities, classes = predict(image, model)
    
print("===========================")
print("Prediction Data")
print("===========================")
print("The Most Likely class:  " + classes[0])
print("---------------------------")      
print("The Probability of this class: {:.4f}".format(probabilities[0]))
print("===========================")  