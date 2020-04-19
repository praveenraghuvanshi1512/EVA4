import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR


def findandplotlearningrate(model, trainloader, criterion, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=momentum)
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100, step_mode="exp")
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    print('Min loss value is : {} \nMin LR value is   : {}'.format(min(lr_finder.history['loss']),format(min(lr_finder.history['lr']),'.10f')))
    
def plotCyclicLR(total_iterations,min_lr,max_lr,step_size):
  l_rate=[]
  for iteration in range(total_iterations):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs((iteration / step_size) - 2 * cycle + 1)
    lr = min_lr + ((max_lr - min_lr) * (1 - x))
    l_rate.append(lr)
  plt.plot(list(range(total_iterations)),l_rate)
  
def performLRRangeTest(model, optimizer, criterion, device, trainloader, end_lr, num_iter):
    lr_finder = LRFinder(model, optimizer, criterion, device)
    lr_finder.range_test(trainloader, end_lr=end_lr, num_iter=num_iter, step_mode="exp")
    lr_finder.plot()
    max_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]
    max_lr = float('{:4f}'.format(max_lr))
    print(max_lr)
    lr_finder.reset()
    return max_lr
    
def getSchdeduler(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.25):
    return OneCycleLR(optimizer, max_lr = max_lr, total_steps=total_steps, epochs=
                       epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start, anneal_strategy='linear', 
                       cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=10.0, final_div_factor=100)
                       
def getlr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    