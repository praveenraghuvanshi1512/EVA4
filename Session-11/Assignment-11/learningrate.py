import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder

def findandplotlearningrate(model, trainloader, criterion, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=momentum)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    print('Min loss value is : {} \nMin LR value is   : {}'.format(min(lr_finder.history['loss']),format(min(lr_finder.history['lr']),'.10f')))