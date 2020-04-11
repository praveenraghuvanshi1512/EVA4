import torch
import torch.nn as nn
import torch.nn.functional as F

class S11Model(nn.Module):
	def __init__(self):
		super(S11Model, self).__init__()
		
		self.preplayer =nn.Sequential(
		nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.BatchNorm2d(64),
		nn.ReLU()
		)
		
		
		self.Layer1= nn.Sequential(
		nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.MaxPool2d(2,2),
		nn.BatchNorm2d(128),
		nn.ReLU()
		)
		
		self.R1= nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.BatchNorm2d(128),
		nn.ReLU(),
		nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.BatchNorm2d(128),
		nn.ReLU()
		)
		
		self.Layer2= nn.Sequential(
		nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.MaxPool2d(2,2),
		nn.BatchNorm2d(256),
		nn.ReLU()
		)
		
		self.Layer3=nn.Sequential(
		nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.MaxPool2d(2,2),
		nn.BatchNorm2d(512),
		nn.ReLU()
		)
		
		self.R2=nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.BatchNorm2d(512),
		nn.ReLU(),
		nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3),bias=False, padding=1, padding_mode='same'),
		nn.BatchNorm2d(512),
		nn.ReLU()
		)
		
		self.fc= nn.Sequential(
		nn.Conv2d(in_channels=512, out_channels=10, kernel_size=(1,1),bias=False)
		)
		self.pool=nn.MaxPool2d(4,2)
		
		
	def forward(self,x):
		x=self.preplayer(x)
		x=self.Layer1(x)
		r1=self.R1(x)
		x=x+r1
		x=self.Layer2(x)
		x=self.Layer3(x)
		r2=self.R2(x)
		x=x+r2
		x=self.pool(x)
		x=self.fc(x)
		x=x.view(-1,10)
		x=F.log_softmax(x)
		return x