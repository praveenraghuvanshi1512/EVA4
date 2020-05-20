import os
import PIL
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import augmentation

from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid, save_image
from utils import progress_bar
from dataset import ImageDataset

import io,glob,os,time,random
from shutil import move
from os.path import join
from os import listdir,rmdir

import scipy.ndimage as nd
import numpy as np

def myfunc():
    abc = 10
    print(abc)

def transformations():

    transform_train = augmentation.AlbumentationTransformTrain()
    transform_test = augmentation.AlbumentationTransformTest()
    
    return (transform_train, transform_test)

def loaddataset(root, transformations):
    dataset = ImageDataset(root, transformations)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle =True, pin_memory=True)
    return (dataset, dataloader)

%matplotlib inline
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model,criterion,device,trainloader,optimizer,epoch):
  model.train()
  pbar = tqdm(trainloader)
  for batch_idx, data in enumerate(pbar):
    data["bg_fg"] = data["bg_fg"].to(device)
    data["mask"] = data["mask"].to(device)
    data["depth"] = data["depth"].to(device)
    
    optimizer.zero_grad()
    output = model(data)
    loss1 = criterion(output[0],data["mask"])
    loss2 = criterion(output[1],data["depth"])
    loss = 2*loss1 + loss2
    print('  loss: {}'.format(loss))
    loss.backward()
    optimizer.step()

    if batch_idx %50 == 0 or epoch %10 == 0:
      print('Train Epoch: {}  [{}/{}  ({:.0f}%)]\tLoss:{:.6f}'.format(
          epoch,batch_idx*len(data),len(trainloader.dataset),
          100.*batch_idx/len(trainloader),loss.item()))

      # Mask
      print("\n\nActual mask")
      show(data['mask'].detach().cpu())
      print("\n\nPredicted mask")
      show(output[0].detach().cpu())

      # Depth
      print("\n\nActual Depth")
      show(data['depth'].detach().cpu())
      print("\n\nPredicted Depth")
      show(output[1].detach().cpu())

      # torch.save(model.state_dict(), SAVE_PATH/f"{epoch}_{batch_idx}_{loss.item()}.pth")
'''
def loadcifar10dataset(transform_train, transform_test):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                         shuffle=True, num_workers=2)

    return (trainset, trainloader, testset, testloader)
    
def loadimagenetdataset(train_dir, test_dir, transform_train, transform_test, batch_size=512):
    trainset = datasets.ImageFolder(train_dir, 
                                    transform=transform_train)
    
    testset = datasets.ImageFolder(test_dir, 
                                    transform=transform_test)
    
    print('Preparing data loaders ...')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2)
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                                    shuffle=True, num_workers=2)
                                                    
    return (trainset, trainloader, testset, testloader)
    
def splittinyimagedataset():
    target_folder = './tiny-imagenet-200/val/'
    dest_folder = './tiny-imagenet-200/train/' 

    val_dict={}

    with open('./tiny-imagenet-200/val/val_annotations.txt','r') as f:
        for line in f.readlines():
            splitline = line.split('\t')
            val_dict[splitline[0]] = splitline[1]
        paths = glob.glob('./tiny-imagenet-200/val/images/*')
      
        for path in paths:
            file = path.split('/')[-1].split('\\')[-1]
            folder = val_dict[file]
            dest = dest_folder + str(folder) + '/images/' + str(file)
            move(path,dest)
            
    target_folder = './tiny-imagenet-200/train/'
    train_folder = './tiny-imagenet-200/train_set/'
    test_folder = './tiny-imagenet-200/test_set/'
     
    os.mkdir(train_folder)
    os.mkdir(test_folder)
     
    paths = glob.glob('./tiny-imagenet-200/train/*')
 
    for path in paths:
        folder = path.split('/')[-1].split('\\')[-1]
        source = target_folder + str(folder + '/images/')
        train_dest = train_folder + str(folder + '/')
        test_dest = test_folder + str(folder + '/')
        os.mkdir(train_dest)
        os.mkdir(test_dest)
        images = glob.glob(source + str('*'))
        #print(len(images))
        # making random
        random.shuffle(images)
      
        test_imgs = images[:165].copy()
        train_imgs = images[165:].copy()
      
        # moving 30% for validation
        for image in test_imgs:
          file = image.split('/')[-1].split('\\')[-1]
          dest = test_dest + str(file)
          move(image, dest)
      
        # moving 70% for training
        for image in train_imgs:
          file = image.split('/')[-1].split('\\')[-1]
          dest = train_dest + str(file)
          move(image, dest)

def getclasses():
    return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def getoptimizer(network, lr, momentum=0.9, nesterov=False, weight_decay=0):
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    
def getscheduler(optimizer):
    return ReduceLROnPlateau(optimizer, mode='min')
    # return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

def getloss():
    return nn.CrossEntropyLoss()
    # return nn.NLLLoss()

def train(network, trainloader, device, optimizer, criterion, trainaccuracies, trainlosses, epoch):
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
        accuracy = 100.*correct/total
        
        progress_bar(batch_idx, len(trainloader), 'Train >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), accuracy, correct, total))
        
        print('Train:: Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    trainaccuracies.append(accuracy)
    train_loss /= len(trainloader.dataset)
    trainlosses.append(train_loss)

def test(network, testloader, device, criterion, valaccuracies, vallosses, epoch):
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
            accuracy = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Test >> Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), accuracy, correct, total))
            
            # scheduler.step(effective_loss)
    
    valaccuracies.append(accuracy)
    test_loss /= len(testloader.dataset)
    vallosses.append(test_loss)
    
    return test_loss

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
    imshow(torchvision.utils.make_grid(images[:noofimages]))

    # display labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(noofimages)))
'''

def show(tensors, figsize=(10, 10), *args, **kwargs):
    grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def draw_and_save(tensors, name, figsize=(20,20), *args, **kwargs):
    try:
        tensors = tensors.detach().cpu()
    except:
        pass
    grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
    grid_image = grid_tensor.permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])

    plt.savefig(name, bbox_inches='tight')
    plt.close()       

