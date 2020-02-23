# EVA4 Assignment 6 - Praveen Raghuvanshi



- [Github Link](https://github.com/praveenraghuvanshi1512/EVA4/tree/Session-5/Session-5/Assignment-5)

- **Best Test Accuracy**: 99.46% (Target)

- **No of parameters**: 9,838

- **No of Epochs**: 15

- **Fully Connected Layer:** No

- ReLU, Batch Normalization, SGD Optimizer, Learning_Rate(0.01), Augmentation(Rotation: +-20), GAP, 1x1

- **Final Model**

  ```python
  '''Neural Network                      Input      Output     RF
    Input Block(Conv1) ->                 28          26        3     3
    Conv Block-1(Conv2) ->                26          24        5     5
    Transition Block-1(Conv3) ->          24          12        10    6
    Conv Block-2(Conv4 -> Conv5) ->       12          8         14    10
    Transition Block-1(Conv3) ->          8           4         28    14
    Output Block(Conv8 -> Conv9)          4           1         ??    
  '''
  class Net(nn.Module):
      def __init__(self):
          super(Net, self).__init__()
  
          # Input Block
          self.convblock1 = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(14)
          ) # output_size = 26
  
          # CONVOLUTION BLOCK 1
          self.convblock2 = nn.Sequential(
              nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16)
          ) # output_size = 24
  
          # TRANSITION BLOCK 1
          self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12
          self.convblock3 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(8)
          ) # output_size = 12
  
          # CONVOLUTION BLOCK 2
          self.convblock4 = nn.Sequential(
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16)
          ) # output_size = 10
          self.convblock5 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(32)
          ) # output_size = 8
  
          # TRANSITION BLOCK 2
          self.pool2 = nn.MaxPool2d(2, 2) # output_size = 4
          self.convblock6 = nn.Sequential(
              nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(8)
          ) # output_size = 4
  
          # OUTPUT BLOCK
          self.convblock7 = nn.Sequential(
              nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(16)
          ) # output_size = 2
  
          self.convblock8 = nn.Sequential(
              nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
              nn.ReLU(),
              nn.BatchNorm2d(10)
          ) # output_size = 2
          self.gap = nn.Sequential(
              nn.AvgPool2d(kernel_size=2)
          ) # output_size = 1
  
      def forward(self, x):
          # Input block
          x = self.convblock1(x)
          # Block-1
          x = self.convblock2(x)
          # Transition Block-1
          x = self.pool1(x)
          x = self.convblock3(x)
          # Block-2        
          x = self.convblock4(x)
          x = self.convblock5(x)
          # Transition Block-2
          x = self.pool2(x)
          x = self.convblock6(x) 
          # Output Block   
          x = self.convblock7(x)  
          x = self.convblock8(x)    
          x = self.gap(x)
          # Reshape
          x = x.view(-1, 10)
          return F.log_softmax(x, dim=-1) # Classification
  ```

- Parameters

  ```html
  Requirement already satisfied: torchsummary in /usr/local/lib/python3.6/dist-packages (1.5.1)
  cuda
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1           [-1, 14, 26, 26]             126
                ReLU-2           [-1, 14, 26, 26]               0
         BatchNorm2d-3           [-1, 14, 26, 26]              28
              Conv2d-4           [-1, 16, 24, 24]           2,016
                ReLU-5           [-1, 16, 24, 24]               0
         BatchNorm2d-6           [-1, 16, 24, 24]              32
           MaxPool2d-7           [-1, 16, 12, 12]               0
              Conv2d-8            [-1, 8, 12, 12]             128
                ReLU-9            [-1, 8, 12, 12]               0
        BatchNorm2d-10            [-1, 8, 12, 12]              16
             Conv2d-11           [-1, 16, 10, 10]           1,152
               ReLU-12           [-1, 16, 10, 10]               0
        BatchNorm2d-13           [-1, 16, 10, 10]              32
             Conv2d-14             [-1, 32, 8, 8]           4,608
               ReLU-15             [-1, 32, 8, 8]               0
        BatchNorm2d-16             [-1, 32, 8, 8]              64
          MaxPool2d-17             [-1, 32, 4, 4]               0
             Conv2d-18              [-1, 8, 4, 4]             256
               ReLU-19              [-1, 8, 4, 4]               0
        BatchNorm2d-20              [-1, 8, 4, 4]              16
             Conv2d-21             [-1, 16, 2, 2]           1,152
               ReLU-22             [-1, 16, 2, 2]               0
        BatchNorm2d-23             [-1, 16, 2, 2]              32
             Conv2d-24             [-1, 10, 2, 2]             160
               ReLU-25             [-1, 10, 2, 2]               0
        BatchNorm2d-26             [-1, 10, 2, 2]              20
          AvgPool2d-27             [-1, 10, 1, 1]               0
  ================================================================
  Total params: 9,838
  Trainable params: 9,838
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.00
  Forward/backward pass size (MB): 0.56
  Params size (MB): 0.04
  Estimated Total Size (MB): 0.60
  ----------------------------------------------------------------
  ```

  

- Logs

  - Highest Accuracy

    ```html
    EPOCH: 12
    Loss=0.04770439863204956 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:13<00:00, 34.08it/s]
    0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0206, Accuracy: 9946/10000 (99.46%)
    ```
  ```
  
  
  
  - Full Logs
  
    ```html
      0%|          | 0/469 [00:00<?, ?it/s]EPOCH: 0
    Loss=0.11708194017410278 Batch_id=468 Accuracy=92.52: 100%|██████████| 469/469 [00:13<00:00, 34.34it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0719, Accuracy: 9853/10000 (98.53%)
    
    EPOCH: 1
    Loss=0.13140122592449188 Batch_id=468 Accuracy=97.53: 100%|██████████| 469/469 [00:14<00:00, 33.30it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0549, Accuracy: 9873/10000 (98.73%)
    
    EPOCH: 2
    Loss=0.04802983999252319 Batch_id=468 Accuracy=98.02: 100%|██████████| 469/469 [00:13<00:00, 34.60it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0356, Accuracy: 9915/10000 (99.15%)
    
    EPOCH: 3
    Loss=0.08869989961385727 Batch_id=468 Accuracy=98.22: 100%|██████████| 469/469 [00:13<00:00, 34.64it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0361, Accuracy: 9912/10000 (99.12%)
    
    EPOCH: 4
    Loss=0.12001284211874008 Batch_id=468 Accuracy=98.34: 100%|██████████| 469/469 [00:13<00:00, 34.84it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0313, Accuracy: 9912/10000 (99.12%)
    
    EPOCH: 5
    Loss=0.0848197415471077 Batch_id=468 Accuracy=98.36: 100%|██████████| 469/469 [00:14<00:00, 32.72it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0281, Accuracy: 9931/10000 (99.31%)
    
    EPOCH: 6
    Loss=0.023646125569939613 Batch_id=468 Accuracy=98.44: 100%|██████████| 469/469 [00:13<00:00, 34.09it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0336, Accuracy: 9905/10000 (99.05%)
    
    EPOCH: 7
    Loss=0.02217532880604267 Batch_id=468 Accuracy=98.56: 100%|██████████| 469/469 [00:13<00:00, 34.40it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0268, Accuracy: 9923/10000 (99.23%)
    
    EPOCH: 8
    Loss=0.04253929480910301 Batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:13<00:00, 34.77it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0234, Accuracy: 9935/10000 (99.35%)
    
    EPOCH: 9
    Loss=0.03841227665543556 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:13<00:00, 33.51it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0257, Accuracy: 9926/10000 (99.26%)
    
    EPOCH: 10
    Loss=0.03174147382378578 Batch_id=468 Accuracy=98.78: 100%|██████████| 469/469 [00:13<00:00, 34.18it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0234, Accuracy: 9943/10000 (99.43%)
    
    EPOCH: 11
    Loss=0.038328707218170166 Batch_id=468 Accuracy=98.75: 100%|██████████| 469/469 [00:13<00:00, 33.99it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0251, Accuracy: 9928/10000 (99.28%)
    
    EPOCH: 12
    Loss=0.04770439863204956 Batch_id=468 Accuracy=98.81: 100%|██████████| 469/469 [00:13<00:00, 34.08it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0206, Accuracy: 9946/10000 (99.46%)
    
    EPOCH: 13
    Loss=0.01455767173320055 Batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:13<00:00, 33.59it/s]
      0%|          | 0/469 [00:00<?, ?it/s]
    Test set: Average loss: 0.0219, Accuracy: 9946/10000 (99.46%)
    
    EPOCH: 14
    Loss=0.14564792811870575 Batch_id=468 Accuracy=98.88: 100%|██████████| 469/469 [00:13<00:00, 34.61it/s]
    
    Test set: Average loss: 0.0225, Accuracy: 9938/10000 (99.38%)
    
  ```



### Summary

| Iteration | Model Change                   | Params | Train Acc | Test Acc | Acc Gap | Overfitting(Max Test Acc) | Disease          | Remedy                 | Analysis                        |
| --------- | ------------------------------ | ------ | --------- | -------- | ------- | ------------------------- | ---------------- | ---------------------- | ------------------------------- |
| 1         | Basic Model                    | 30544  | 99.08     | 98.90    | 0.18    | No(99.82)                 | No of parameters | Reduce channels        | Model is ok. Acc                |
| 2         | Lighter + Batch Normalization  | 7612   | 99.19     | 98.98    | 0.21    | No(99.79)                 | Less accuracy    | Increase Capacity      | Model is good. Used GAP         |
| 3         | Increase Capacity              | 8908   | 99.55     | 99.25    | 0.30    | No(99.70)                 | Less accuracy    | Add Image augmentation | Good Model, have capacity       |
| 4         | Augmentation. Rotation : +- 20 | 8908   | 98.82     | 99.39    | -0.57   | No(100.57)                | Less accuracy    | Add LR and capacity    | Acc improved                    |
| 5         | LR + Increase Capacity         | 9838   | 98.88     | 99.46    | -0.58   | No(100.58)                | --               | --                     | Target Acc achieved in 3 epochs |



## [Code 1: BASE Model](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-5/Session-5/Assignment-5/EVA_4_S5_Praveen_Raghuvanshi_Base.ipynb)

**Target:**

1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set Basic Working Code
5. Set Basic Training & Test Loop
6. Results:
   1. Parameters: 30,544
   2. Best Training Accuracy: 99.16
   3. Best Test Accuracy: 98.82
7. Analysis:
   1. Little Heavy Model for such a problem
   2. No overfitting



## [Code 2: LIGHTER and Batch Norm Model](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-5/Session-5/Assignment-5/EVA_4_S5_Praveen_Raghuvanshi_LIGHT_BatchNorm.ipynb)

**Target:**

1. Make the model lighter
2. Add Batch norm to increase efficiency
3. Results:
   1. Parameters: 7,612
   2. Best Training Accuracy: 99.19
   3. Best Test Accuracy: 98.98
4. Analysis:
   1. Reduced Channels
   2. Used GAP
   3. Acc improved
   4. No overfitting



## [Code 3: Increase capacity](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-5/Session-5/Assignment-5/EVA_4_S5_Praveen_Raghuvanshi_INCCAPACITY.ipynb)

**Target:**

1. Add more layers to increase capacity
2. Manipulated channels
3. Results:
   1. Parameters: 8908
   2. Best Training Accuracy: 99.55
   3. Best Test Accuracy: 99.25
4. Analysis:
   1. Acc improved
   2. No overfitting

## [Code 4: Augmentation](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-5/Session-5/Assignment-5/EVA_4_S5_Praveen_Raghuvanshi_Augmentation.ipynb)

**Target:**

1. Added rotation augmentation(+-20)
2. Results:
   1. Parameters: 8908
   2. Best Training Accuracy: 98.82
   3. Best Test Accuracy: 99.39
3. Analysis:
   1. Data augmentation increased the data and Acc improved
   2. No overfitting

## [Code 5: LR + Increase Capacity](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-5/Session-5/Assignment-5/EVA_4_S5_Praveen_Raghuvanshi_LR_INCCAPACITY.ipynb)

**Target:**

1. Added LR
2. Increase Capacity
3. Results:
   1. Parameters: 9838
   2. Best Training Accuracy: 98.88
   3. Best Test Accuracy: 99.46
4. Analysis:
   1. Improved accuracy and target achieved with accuracy > 99.4 in 3 epochs.
   2. No overfitting