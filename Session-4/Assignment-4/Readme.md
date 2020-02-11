# EVA4 Assignment 4 - Praveen Raghuvanshi

### Team

- Praveen Raghuvanshi - praveenraghuvanshi@gmail.com

- Gowtham Kumar - Kumar.gowtham@gmail.com

- Rohit - rohitfattepur@gmail.com

- Veera - infochunduri@gmail.com

  

## Assignment

- [Github Link](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-4/Session-4/Assignment-4/EVA_4_Assignment_4_Praveen_Raghuvanshi.ipynb)

- [Colab Link](https://colab.research.google.com/drive/1WDebiK-hB0isRslHRL8S0ixTiNBeQt5k?authuser=1#scrollTo=9dAn_w-kQcaA)

- [Solution File(ipynb)](EVA_4_Assignment_4_Praveen_Raghuvanshi.ipynb)

- **Validation Accuracy**: 98.8%

- **No of parameters**: 14,290

- **No of Epochs**: 20

- **Fully Connected Layer:** No

- ReLU, Batch Normalization

- Model

  ```python
  class Model(nn.Module):
      def __init__(self):
          super(Model, self).__init__()
          self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
          self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
          self.pool1 = nn.MaxPool2d(2, 2)
          self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
          self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
          self.pool2 = nn.MaxPool2d(2, 2)
          self.conv5 = nn.Conv2d(16, 8, 3)
          self.conv6 = nn.Conv2d(8, 16, 3)
          self.conv7 = nn.Conv2d(16, 10, 3)
  
      def forward(self, x):
          x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
          x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
          x = F.relu(self.conv6(F.relu(self.conv5(x))))
          x = self.conv7(x)
          x = x.view(-1, 10)
          return F.log_softmax(x)
  ```

- Parameters

  ```html
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1            [-1, 8, 28, 28]              80
              Conv2d-2           [-1, 16, 28, 28]           1,168
           MaxPool2d-3           [-1, 16, 14, 14]               0
              Conv2d-4           [-1, 32, 14, 14]           4,640
              Conv2d-5           [-1, 16, 14, 14]           4,624
           MaxPool2d-6             [-1, 16, 7, 7]               0
              Conv2d-7              [-1, 8, 5, 5]           1,160
              Conv2d-8             [-1, 16, 3, 3]           1,168
              Conv2d-9             [-1, 10, 1, 1]           1,450
  ================================================================
  Total params: 14,290
  Trainable params: 14,290
  Non-trainable params: 0
  ----------------------------------------------------------------
  ```

  

- Logs

  ```html
    0%|          | 0/20 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:20: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
    5%|▌         | 1/20 [00:14<04:33, 14.37s/it]Epoch: 1/20..  Time: 14.36s.. Training Loss: 0.796..  Training Accu: 0.745..  Val Loss: 0.284..  Val Accu: 0.915
   10%|█         | 2/20 [00:28<04:17, 14.29s/it]Epoch: 2/20..  Time: 14.09s.. Training Loss: 0.236..  Training Accu: 0.929..  Val Loss: 0.167..  Val Accu: 0.948
   15%|█▌        | 3/20 [00:42<04:02, 14.25s/it]Epoch: 3/20..  Time: 14.15s.. Training Loss: 0.160..  Training Accu: 0.951..  Val Loss: 0.126..  Val Accu: 0.961
   20%|██        | 4/20 [00:56<03:46, 14.16s/it]Epoch: 4/20..  Time: 13.96s.. Training Loss: 0.127..  Training Accu: 0.961..  Val Loss: 0.102..  Val Accu: 0.968
   25%|██▌       | 5/20 [01:11<03:33, 14.24s/it]Epoch: 5/20..  Time: 14.44s.. Training Loss: 0.109..  Training Accu: 0.966..  Val Loss: 0.092..  Val Accu: 0.971
   30%|███       | 6/20 [01:24<03:18, 14.16s/it]Epoch: 6/20..  Time: 13.98s.. Training Loss: 0.096..  Training Accu: 0.970..  Val Loss: 0.079..  Val Accu: 0.975
   35%|███▌      | 7/20 [01:38<03:03, 14.08s/it]Epoch: 7/20..  Time: 13.88s.. Training Loss: 0.087..  Training Accu: 0.974..  Val Loss: 0.076..  Val Accu: 0.975
   40%|████      | 8/20 [01:52<02:48, 14.06s/it]Epoch: 8/20..  Time: 13.99s.. Training Loss: 0.081..  Training Accu: 0.975..  Val Loss: 0.076..  Val Accu: 0.975
  Validation loss has not improved since: 0.076.. Count:  1
   45%|████▌     | 9/20 [02:06<02:34, 14.03s/it]Epoch: 9/20..  Time: 13.98s.. Training Loss: 0.076..  Training Accu: 0.977..  Val Loss: 0.058..  Val Accu: 0.982
   50%|█████     | 10/20 [02:21<02:20, 14.08s/it]Epoch: 10/20..  Time: 14.17s.. Training Loss: 0.071..  Training Accu: 0.978..  Val Loss: 0.061..  Val Accu: 0.980
  Validation loss has not improved since: 0.058.. Count:  1
   55%|█████▌    | 11/20 [02:34<02:06, 14.03s/it]Epoch: 11/20..  Time: 13.92s.. Training Loss: 0.068..  Training Accu: 0.979..  Val Loss: 0.059..  Val Accu: 0.981
  Validation loss has not improved since: 0.058.. Count:  2
   60%|██████    | 12/20 [02:49<01:52, 14.08s/it]Epoch: 12/20..  Time: 14.19s.. Training Loss: 0.065..  Training Accu: 0.980..  Val Loss: 0.052..  Val Accu: 0.984
   65%|██████▌   | 13/20 [03:02<01:38, 14.00s/it]Epoch: 13/20..  Time: 13.82s.. Training Loss: 0.062..  Training Accu: 0.981..  Val Loss: 0.051..  Val Accu: 0.984
   70%|███████   | 14/20 [03:17<01:24, 14.05s/it]Epoch: 14/20..  Time: 14.15s.. Training Loss: 0.059..  Training Accu: 0.982..  Val Loss: 0.050..  Val Accu: 0.984
   75%|███████▌  | 15/20 [03:31<01:10, 14.01s/it]Epoch: 15/20..  Time: 13.94s.. Training Loss: 0.057..  Training Accu: 0.982..  Val Loss: 0.055..  Val Accu: 0.982
  Validation loss has not improved since: 0.050.. Count:  1
   80%|████████  | 16/20 [03:45<00:56, 14.06s/it]Epoch: 16/20..  Time: 14.15s.. Training Loss: 0.056..  Training Accu: 0.983..  Val Loss: 0.046..  Val Accu: 0.984
   85%|████████▌ | 17/20 [03:59<00:41, 14.00s/it]Epoch: 17/20..  Time: 13.87s.. Training Loss: 0.054..  Training Accu: 0.984..  Val Loss: 0.048..  Val Accu: 0.985
  Validation loss has not improved since: 0.046.. Count:  1
   90%|█████████ | 18/20 [04:13<00:28, 14.06s/it]Epoch: 18/20..  Time: 14.18s.. Training Loss: 0.051..  Training Accu: 0.984..  Val Loss: 0.047..  Val Accu: 0.985
  Validation loss has not improved since: 0.046.. Count:  2
   95%|█████████▌| 19/20 [04:27<00:14, 14.05s/it]Epoch: 19/20..  Time: 14.03s.. Training Loss: 0.050..  Training Accu: 0.985..  Val Loss: 0.048..  Val Accu: 0.985
  Validation loss has not improved since: 0.046.. Count:  3
  100%|██████████| 20/20 [04:41<00:00, 13.99s/it]Epoch: 20/20..  Time: 13.85s.. Training Loss: 0.048..  Training Accu: 0.985..  Val Loss: 0.041..  Val Accu: 0.988
  
  ```

  

#### Analysis

- Dataset: MNIST

- Image size: 28 x 28 x 1

- Train data: 60000, Test data: 10000

- Classes (10) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

- Images are of same size, centered and size normalized

- MNIST is a labeled dataset and falls under supervised learning

  