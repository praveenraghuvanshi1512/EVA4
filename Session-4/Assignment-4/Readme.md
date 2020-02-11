# EVA4 Assignment 4 - Praveen Raghuvanshi

### Team

- **Praveen Raghuvanshi** - praveenraghuvanshi@gmail.com

- Gowtham Kumar - Kumar.gowtham@gmail.com

- Rohit - rohitfattepur@gmail.com

- Veera - infochunduri@gmail.com

  

## Assignment

- [Github Link](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-4/Session-4/Assignment-4/EVA_4_Assignment_4_Praveen_Raghuvanshi.ipynb)

- [Colab Link](https://colab.research.google.com/drive/1WDebiK-hB0isRslHRL8S0ixTiNBeQt5k?authuser=1#scrollTo=9dAn_w-kQcaA)

- [Solution File(ipynb)](EVA_4_Assignment_4_Praveen_Raghuvanshi.ipynb)

- **Validation Accuracy**: 99.4% (Target)

- **No of parameters**: 19,280

- **No of Epochs**: 20

- **Fully Connected Layer:** No

- ReLU, Batch Normalization, Adam Optimizer, Dropout(0.1), Batch_Size(32), Learning_Rate(1e-4)

- Model

  ```python
  class Model(nn.Module):
      def __init__(self):
          super(Model, self).__init__()
          self.conv1 = nn.Conv2d(1, 16, 3, bias=False)  # Input - 28, Output - 26, RF  - 3
          self.bn1   = nn.BatchNorm2d(16)
          self.conv2 = nn.Conv2d(16, 32, 3, bias=False) # Input - 26, Output - 24, RF  - 5
          self.bn2   = nn.BatchNorm2d(32)
          self.pool1 = nn.MaxPool2d(2, 2)               # Input - 24, Output - 12, RF  - 10
          self.conv3 = nn.Conv2d(32, 16, 3, bias=False) # Input - 12, Output - 10, RF  - 12
          self.bn3   = nn.BatchNorm2d(16)
          self.conv4 = nn.Conv2d(16, 32, 3, bias=False) # Input - 10, Output - 8, RF  - 14
          self.bn4   = nn.BatchNorm2d(32)
          self.pool2 = nn.MaxPool2d(2, 2)               # Input - 8, Output - 4, RF  - 28
          self.conv5 = nn.Conv2d(32, 10, 4, bias=False) # Input - 4, Output - 2, RF  - 30        
          self.dropout = nn.Dropout2d(0.1)
          
      def forward(self, x):
          x = self.pool1(self.dropout(F.relu(self.bn2(self.conv2(self.dropout(F.relu(self.bn1(self.conv1(x)))))))))
          x = self.pool2(self.dropout(F.relu(self.bn4(self.conv4(self.dropout(F.relu(self.bn3(self.conv3(x)))))))))
          x = self.conv5(x)
          x = x.view(-1, 10)
          return F.log_softmax(x)
  ```

- Parameters

  ```html
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1           [-1, 16, 26, 26]             144
         BatchNorm2d-2           [-1, 16, 26, 26]              32
           Dropout2d-3           [-1, 16, 26, 26]               0
              Conv2d-4           [-1, 32, 24, 24]           4,608
         BatchNorm2d-5           [-1, 32, 24, 24]              64
           Dropout2d-6           [-1, 32, 24, 24]               0
           MaxPool2d-7           [-1, 32, 12, 12]               0
              Conv2d-8           [-1, 16, 10, 10]           4,608
         BatchNorm2d-9           [-1, 16, 10, 10]              32
          Dropout2d-10           [-1, 16, 10, 10]               0
             Conv2d-11             [-1, 32, 8, 8]           4,608
        BatchNorm2d-12             [-1, 32, 8, 8]              64
          Dropout2d-13             [-1, 32, 8, 8]               0
          MaxPool2d-14             [-1, 32, 4, 4]               0
             Conv2d-15             [-1, 10, 1, 1]           5,120
  ================================================================
  Total params: 19,280
  Trainable params: 19,280
  Non-trainable params: 0
  ----------------------------------------------------------------
  ```

  

- Logs

  - Highest Accuracy

    ```html
    100%|██████████| 20/20 [05:39<00:00, 17.03s/it]Epoch: 20/20..  Time: 16.88s.. Training Loss: 0.029..  Training Accu: 0.991..  Val Loss: 0.020..  Val Accu: 0.994
    ```

    

  - Full Logs

    ```html
      0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:22: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
      5%|▌         | 1/20 [00:17<05:38, 17.84s/it]Epoch: 1/20..  Time: 17.84s.. Training Loss: 0.538..  Training Accu: 0.869..  Val Loss: 0.129..  Val Accu: 0.967
     10%|█         | 2/20 [00:34<05:14, 17.47s/it]Epoch: 2/20..  Time: 16.59s.. Training Loss: 0.139..  Training Accu: 0.964..  Val Loss: 0.074..  Val Accu: 0.979
     15%|█▌        | 3/20 [00:51<04:53, 17.28s/it]Epoch: 3/20..  Time: 16.84s.. Training Loss: 0.096..  Training Accu: 0.973..  Val Loss: 0.057..  Val Accu: 0.984
     20%|██        | 4/20 [01:08<04:34, 17.15s/it]Epoch: 4/20..  Time: 16.83s.. Training Loss: 0.076..  Training Accu: 0.978..  Val Loss: 0.044..  Val Accu: 0.987
     25%|██▌       | 5/20 [01:24<04:15, 17.05s/it]Epoch: 5/20..  Time: 16.84s.. Training Loss: 0.067..  Training Accu: 0.980..  Val Loss: 0.040..  Val Accu: 0.987
     30%|███       | 6/20 [01:42<04:00, 17.15s/it]Epoch: 6/20..  Time: 17.38s.. Training Loss: 0.059..  Training Accu: 0.983..  Val Loss: 0.036..  Val Accu: 0.988
     35%|███▌      | 7/20 [01:59<03:41, 17.03s/it]Epoch: 7/20..  Time: 16.74s.. Training Loss: 0.053..  Training Accu: 0.984..  Val Loss: 0.034..  Val Accu: 0.990
     40%|████      | 8/20 [02:15<03:23, 16.97s/it]Epoch: 8/20..  Time: 16.82s.. Training Loss: 0.050..  Training Accu: 0.985..  Val Loss: 0.031..  Val Accu: 0.990
     45%|████▌     | 9/20 [02:33<03:07, 17.04s/it]Epoch: 9/20..  Time: 17.22s.. Training Loss: 0.047..  Training Accu: 0.986..  Val Loss: 0.028..  Val Accu: 0.990
     50%|█████     | 10/20 [02:49<02:49, 16.97s/it]Epoch: 10/20..  Time: 16.80s.. Training Loss: 0.044..  Training Accu: 0.987..  Val Loss: 0.027..  Val Accu: 0.991
     55%|█████▌    | 11/20 [03:06<02:31, 16.78s/it]Epoch: 11/20..  Time: 16.34s.. Training Loss: 0.042..  Training Accu: 0.988..  Val Loss: 0.027..  Val Accu: 0.992
     60%|██████    | 12/20 [03:23<02:14, 16.87s/it]Epoch: 12/20..  Time: 17.07s.. Training Loss: 0.039..  Training Accu: 0.988..  Val Loss: 0.026..  Val Accu: 0.991
     65%|██████▌   | 13/20 [03:40<01:59, 17.02s/it]Epoch: 13/20..  Time: 17.37s.. Training Loss: 0.037..  Training Accu: 0.989..  Val Loss: 0.024..  Val Accu: 0.992
     70%|███████   | 14/20 [03:57<01:41, 16.86s/it]Epoch: 14/20..  Time: 16.48s.. Training Loss: 0.036..  Training Accu: 0.989..  Val Loss: 0.023..  Val Accu: 0.993
     75%|███████▌  | 15/20 [04:14<01:24, 16.93s/it]Epoch: 15/20..  Time: 17.08s.. Training Loss: 0.034..  Training Accu: 0.989..  Val Loss: 0.024..  Val Accu: 0.993
    Validation loss has not improved since: 0.023.. Count:  1
     80%|████████  | 16/20 [04:31<01:07, 16.93s/it]Epoch: 16/20..  Time: 16.94s.. Training Loss: 0.033..  Training Accu: 0.990..  Val Loss: 0.022..  Val Accu: 0.993
     85%|████████▌ | 17/20 [04:48<00:50, 16.89s/it]Epoch: 17/20..  Time: 16.80s.. Training Loss: 0.032..  Training Accu: 0.990..  Val Loss: 0.024..  Val Accu: 0.993
    Validation loss has not improved since: 0.022.. Count:  1
     90%|█████████ | 18/20 [05:04<00:33, 16.88s/it]Epoch: 18/20..  Time: 16.85s.. Training Loss: 0.030..  Training Accu: 0.991..  Val Loss: 0.021..  Val Accu: 0.993
     95%|█████████▌| 19/20 [05:22<00:17, 17.09s/it]Epoch: 19/20..  Time: 17.59s.. Training Loss: 0.030..  Training Accu: 0.991..  Val Loss: 0.021..  Val Accu: 0.993
    100%|██████████| 20/20 [05:39<00:00, 17.03s/it]Epoch: 20/20..  Time: 16.88s.. Training Loss: 0.029..  Training Accu: 0.991..  Val Loss: 0.020..  Val Accu: 0.994
    
    ```

    

  

#### Analysis

- Dataset: MNIST

- Image size: 28 x 28 x 1

- Train data: 60000, Test data: 10000

- Classes (10) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

- Images are of same size, centered and size normalized

- MNIST is a labeled dataset and falls under supervised learning

  