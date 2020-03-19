import os
import PIL
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import augmentation
from utils import progress_bar
from torchvision.utils import make_grid, save_image

def myfunc():
    abc = 10
    print(abc)

def transformations():

    transform_train = augmentation.AlbumentationTransformTrain()
    transform_test = augmentation.AlbumentationTransformTest()
    
    return (transform_train, transform_test)


def loadcifar10dataset(transform_train, transform_test):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=True, num_workers=2)

    return (trainset, trainloader, testset, testloader)

def getclasses():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def getoptimizer(network):
    return optim.Adam(network.parameters(), lr=0.001)
    # return optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

def train(network, trainloader, device, optimizer, criterion, epoch):
    print('\nEpoch: %d' % epoch)
    network.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        '''print('Train:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))'''
        
def test(network, testloader, device, criterion, epoch):
    network.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Test >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def loadimage(imagedirectory, imagename):
    # Load Image
    imgpath = os.path.join(imagedirectory, imagename)
    pilimg = PIL.Image.open(imgpath)
    return pilimg
    
def saveimage(images, outputdirectory, imagename):
    os.makedirs(outputdirectory, exist_ok=True)
    outputname = imagename
    outputpath = os.path.join(outputdirectory, outputname)
    save_image(images, outputpath)
    pilimg = PIL.Image.open(outputpath)
    return pilimg

def display(noofimages, trainloader, classes):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    maximagescount = labels.size()[0]
    if noofimages > maximagescount:
        raise ValueError('The no of images must be less than ' + maximagescount)

    # show images
    imshow(torchvision.utils.make_grid(images))

    # display labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(noofimages)))
