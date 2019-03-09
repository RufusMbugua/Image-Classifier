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
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


arch = {"vgg16":25088,
        "densenet121":1024,
        "alexnet":9216}

    
# Load Flower Data Directories    
def load_data(where  = "./flowers" ):

    data_dir = './flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Apply the required transfomations to the test dataset in order to maximize the efficiency of the learning
    #process


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    validation_transforms = test_transforms

    data_transforms = {'train':train_transforms,'test':test_transforms,'validation':validation_transforms}

    # print(data_transforms)

    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    validation_data = datasets.ImageFolder(valid_dir, transform=data_transforms['validation'])

    image_datasets = {'train':train_data,'test':test_data,'validation':validation_data}

    # print(image_datasets)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    validloader = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=32)

    return trainloader, testloader, validloader, train_data, test_data, validation_data, train_dir ,test_dir, valid_dir

# Create Model
def create_model(structure, dropout, learning_rate, output_labels=102):
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif structure == 'alexnet':
        model = models.alexnet(pretrained = True)
    else:
        print("Im sorry but {} is not a valid model.Try using vgg16,densenet121,or alexnet?".format(structure))
    
    # Create Classifier
    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(arch[structure], 512)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(512, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,output_labels)),
            ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate )

    
    model.cuda()

    return model, criterion, optimizer

# Train the model
def train_model(model, trainloader, validloader, criterion, optimizer, epochs=4, print_every=10, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    loss_show=[]

   
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(validloader):
                    optimizer.zero_grad()

                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():    
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(validloader)
                accuracy = accuracy /len(validloader)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(vlost),
                       "Accuracy: {:.4f}".format(accuracy))


                running_loss = 0

# Confirm Accuracy of the Model
def check_accuracy(model, testloader):    
    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Save the Model to a Checkpoint    
def save_checkpoint( model, train_data, epochs, optimizer, path = 'checkpoint.pth' ):
    checkpoint = {'state_dict': model.state_dict(),
              'classifier': model.classifier,
              'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict,
              'num_epochs': epochs}

    torch.save(checkpoint, path)

# Load the Model from a Checkpoint
def load_checkpoint(model, path = 'checkpoint.pth'):
    
    # Checkpoint for when using GPU
    checkpoint = torch.load(path)   
#     print(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    model.classifier = checkpoint['classifier'] 
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Process Image
def process_image(image):

    image = Image.open(image)
    
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_image = image_transforms(image)
    
    return processed_image

# Show Image on Canvas
def image_show(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array(image)
    image = image.transpose((1,2,0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
