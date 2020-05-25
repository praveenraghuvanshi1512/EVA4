import os
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import augmentation

from torchvision import datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid, save_image
#from utils import progress_bar
from dataset import ImageDataset
from tqdm import tqdm

import io,glob,os,time,random
from shutil import move
from os.path import join
from os import listdir,rmdir

import scipy.ndimage as nd
import numpy as np

import time
import copy
from collections import defaultdict

def myfunc():
    abc = 10
    print(abc)
    
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce # * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
    
def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def transformations():

    transform_train = augmentation.AlbumentationTransformTrain()
    transform_test = augmentation.AlbumentationTransformTest()
    
    return (transform_train, transform_test)

def loaddataset(root, transformations, batch_size=32):
    dataset = ImageDataset(root, transformations)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    return (train_loader, test_loader)

def train(model,criterion,device,trainloader,optimizer,epoch):
  model.train()
  pbar = tqdm(trainloader)

  for batch_idx, data in enumerate(pbar):
    bgfgs = data["bg_fg"].to(device)
    gt_mask = data["mask"].to(device)
    gt_depth = data["depth"].to(device)
    
    optimizer.zero_grad()
    
    output = model(bgfgs)
    pred_mask = output[0]
    pred_depth = output[1]
    
    loss1 = criterion(pred_mask,gt_mask)
    loss2 = criterion(pred_depth,gt_depth)
    loss = 2*loss1 + loss2
    loss.backward()
    
    optimizer.step()

    if batch_idx+1 %100 == 0:
      print('\nTrain Epoch: {}  [{}/{}  ({:.0f}%)]\tLoss:{:.6f}'.format(
          epoch,batch_idx*len(data),len(trainloader.dataset),
          100.*batch_idx/len(trainloader),loss.item()))

      # Mask
      print("\n\nActual mask")
      show(gt_mask.detach().cpu())
      print("\n\nPredicted mask")
      show(pred_mask.detach().cpu())

      # Depth
      print("\n\nActual Depth")
      show(gt_depth.detach().cpu())
      print("\n\nPredicted Depth")
      show(pred_depth.detach().cpu())

      torch.save(model.state_dict(), SAVE_PATH/f"{epoch}_{batch_idx}_{loss.item()}.pth")
      
def train_test_model(start_epoch, num_epochs, valid_loss_min_input, model, criterion, device, trainloader, testloader, optimizer, scheduler=None, save_path='.'):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_train_loss = 1e10
    
    # initialize tracker for minimum validation loss
    valid_loss_min = valid_loss_min_input
    train_loss = []
    valid_loss = []
    
    for epoch in range(start_epoch, start_epoch + num_epochs + 1):
        epochlr = getlr(optimizer)
        print('>>>>>>>> \nEpoch {}/{}, LR: {}'.format(epoch, start_epoch + num_epochs - 1, epochlr))
        t_loss = train_model(model, criterion, device, trainloader, optimizer, epoch, save_path)
        train_loss.append(t_loss)

        v_loss = test_model(model, criterion, device, testloader, epoch, save_path)
        valid_loss.append(v_loss)
        
        if scheduler is not None:
            scheduler.step(v_loss)

        # print training/validation statistics 
        print('************* \nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n'.format(
            epoch, 
            t_loss,
            v_loss
            ))

        ## TODO: save the model if validation loss has decreased
        if v_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,v_loss))
            # save checkpoint as best model
            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': v_loss,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            full_best_path = '{}/best_model.pth'.format(save_path)
            savecheckpoint(checkpoint, full_best_path)
            valid_loss_min = v_loss
            
    return model, train_loss, valid_loss

def train_model(model, criterion, device, trainloader, optimizer, epoch, save_path=''):
    best_loss = 1e10
    metrics = defaultdict(float)
    epoch_samples = 0
    isPrinted = False
    model.train()
    for batch_index, data in enumerate(trainloader):
        bgfgs = data["bg_fg"].to(device)
        gt_mask = data["mask"].to(device)
        gt_depth = data["depth"].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(bgfgs)
        pred_mask = outputs[0]
        pred_depth = outputs[1]
        
        loss_mask = calc_loss(pred_mask, gt_mask, metrics) # criterion(pred_mask,gt_mask)
        loss_depth = calc_loss(pred_depth, gt_depth, metrics) # criterion(pred_depth,gt_depth)
        loss = loss_mask + 2 * loss_depth
        
        loss.backward()
        optimizer.step()
        
        epoch_samples += bgfgs.size(0)
        epoch_loss = metrics['loss'] / epoch_samples

        # deep copy the model
        if epoch_loss < best_loss:
            # print('Best loss at index {}'.format(batch_index))
            # print_metrics(metrics, epoch_samples, 'train')
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        if batch_index%50 == 0:
            print_metrics(metrics, epoch_samples, 'train')
            
        if (epoch == 1 or epoch%4 == 0) and isPrinted == False:
            # Mask
            print("\n\nTrain - Actual mask")
            plotandsave(gt_mask.detach().cpu().narrow(0,0,5), name='{}/train_actual_mask.png'.format(save_path))
            print("\n\nTrain - Predicted mask")
            plotandsave(pred_mask.detach().cpu().narrow(0,0,5), name='{}/train_predicted_mask.png'.format(save_path))

            # Depth
            print("\n\nTrain - Actual Depth")
            plotandsave(gt_depth.detach().cpu().narrow(0,0,5), name='{}/train_actual_depth.png'.format(save_path))
            print("\n\nTrain - Predicted Depth")
            plotandsave(pred_depth.detach().cpu().narrow(0,0,5), name='{}/train_predicted_depth.png'.format(save_path))
            isPrinted = True
            print('Best val loss: {:4f}'.format(best_loss))

    return best_loss
    
    

def test_model(model, criterion, device, testloader, epoch, save_path='.'):
    since = time.time()
    metrics = defaultdict(float)
    epoch_samples = 0
    model.eval()
    isPrinted = False
    for batch_index, data in enumerate(testloader):
        bgfgs = data["bg_fg"].to(device)
        gt_mask = data["mask"].to(device)
        gt_depth = data["depth"].to(device)
        
        outputs = model(bgfgs)
        pred_mask = outputs[0]
        pred_depth = outputs[1]
        
        loss1 = calc_loss(pred_mask, gt_mask, metrics) # criterion(pred_mask,gt_mask)
        loss2 = calc_loss(pred_depth, gt_depth, metrics) # criterion(pred_depth,gt_depth)
        test_loss = 2 * loss1 + loss2
        
        epoch_samples += bgfgs.size(0)
        epoch_loss = metrics['loss'] / epoch_samples
        
        if batch_index % 50 == 0:
            print_metrics(metrics, epoch_samples, 'test')

        if (epoch == 1 or epoch%4 == 0) and isPrinted == False:
            # Mask
            print("\n\nTest - Actual mask: {}".format(epoch))
            plotandsave(gt_mask.detach().cpu().narrow(0,0,5), name='{}/test_actual_mask.png'.format(save_path))
            print("\n\nTest - Predicted mask")
            plotandsave(pred_mask.detach().cpu().narrow(0,0,5), name='{}/test_predicted_mask.png'.format(save_path))

            # Depth
            print("\n\nTest - Actual Depth")
            plotandsave(gt_depth.detach().cpu().narrow(0,0,5), name='{}/test_actual_depth.png'.format(save_path))
            print("\n\nTest - Predicted Depth")
            plotandsave(pred_depth.detach().cpu().narrow(0,0,5), name='{}/test_predicted_depth.png'.format(save_path))
            isPrinted = True
    
    return epoch_loss
        

def show(tensors, figsize=(10, 10), *args, **kwargs):
    grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
    grid_image = grid_tensor.permute(1,2,0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plotandsave(tensors, name, figsize=(10,10), *args, **kwargs):
    grid_tensor = torchvision.utils.make_grid(tensors, *args, **kwargs)
    grid_image = grid_tensor.permute(1, 2, 0)
    plt.figure(figsize=figsize)
    plt.imshow(grid_image)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.savefig(name, bbox_inches='tight')
    plt.close() 
    

def savecheckpoint(state, checkpoint_path):
    torch.save(state, checkpoint_path)
    print('Checkpoint saved: {}'.format(checkpoint_path))
    
def loadcheckpoint(checkpoint_path, model, optimizer, isTrain = True):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    
    if(isTrain == False):
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.eval()
    
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def plotmetrics(trainlosses, testlosses, save_file):
    print(len(trainlosses))
    print(len(trainlosses))
    
    # summarize history for loss
    plt.plot(trainlosses)
    plt.plot(testlosses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.savefig(save_file)    

def getlr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
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
