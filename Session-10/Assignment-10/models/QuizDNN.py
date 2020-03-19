import torch.nn as nn
import torch.nn.functional as F

class QuizDNN(nn.Module):
    def __init__(self):
        super(QuizDNN, self).__init__()
        
        dropout_value = 0.05
        
        # Input Block
        self.convblock1_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        self.convblock1_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        self.convblock1_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2, 2)
        self.convblock2_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        self.convblock2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )

        self.convblock2_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dropout_value)
        )

        self.convblock2_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(dropout_value)
        ) 

        # TRANSITION BLOCK 1
        self.pool2 = nn.MaxPool2d(2, 2) 
        self.convblock3_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        ) 

        self.convblock3_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        self.convblock3_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.convblock3_4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        self.pool3 = nn.MaxPool2d(2, 2)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) 

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        ) 

    def forward(self, x):
        x1 = self.convblock1_1(x) # input conv
        x2 = self.convblock1_2(x1)
        x2 = x1+x2
        x3 = self.convblock1_3(x2)
        x3 = x2+x3
        x4 = self.pool1(x3)
        x5 = self.convblock2_1(x4)
        x4_1 = self.convblock2_2(x4)
        x5 = x4_1+x5
        x6 = self.convblock2_3(x5)
        x6 = x5+x6
        x7 = self.convblock2_4(x6)
        x8 = self.pool2(x7)
        x8_1 = self.convblock3_1(x8)
        x9 = self.convblock3_2(x8)
        x9 = x8_1+x9
        x10 = self.convblock3_3(x9)
        x10 = x9+x10
        x11 = self.convblock3_4(x10)
        x12 = F.adaptive_avg_pool2d(x11, 1)        
        y = self.convblock4(x12)
        y = y.view(-1, 10)
        return F.log_softmax(y, dim=-1)