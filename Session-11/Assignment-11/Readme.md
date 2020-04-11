# EVA4 Assignment 10 - Praveen Raghuvanshi



- [Github link](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-10/Session-10/Assignment-10/EVA_4_S10_Praveen_Raghuvanshi_Main.ipynb)

  - [Assignment Directory](https://github.com/praveenraghuvanshi1512/EVA4/tree/Session-11/Session-11/Assignment-11)
  - [Notebook](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-11/Session-11/Assignment-11/EVA_4_S11_Praveen_Raghuvanshi_Main_91_04.ipynb)

- [Colab link](https://colab.research.google.com/drive/195q1qkmfsgV26bh3kZUz2O13MjDCTbnR#scrollTo=qylKg0Cvq-L9)

- Model: Custom [S11Model](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-11/Session-11/Assignment-11/models/s11model.py)

- No of parameters: 6,573,120

- No of Epochs : 24

- Best Test Acc: 91.04%

- Code Modularity

  - **EVA-4-S11-Praveen-Raghuvanshi-Main-91-04.ipynb** : It contains main workflow
  - **S11_functions.py**: It contains all the functions related to setup, train, load dataset etc.
  - **models/s11model.py** : It contains Custom S11 Model 
  - **utils.py** : It contains helper functions such as progress bar
  - **augmentation.py**: It contains image augmentations such as albumentations
  - **gradcam/gradcam.py**: GradCAM and GRADCAM++ implementation
  - **gradcam/gradcam_utils.py**: Utitlity functions used in GradCAM such as visual_cam
  - Folders:
    - sampleimages: Images from internet on which GradCAM is applied
    - outputs: Outcome of GradCAM

- Logs

  - Best accuracy

    ```python
    
    EPOCH: 24 LR: 0.0012499999999999994
    
    Epoch: 24
     [================================================================>]  Step: 157ms | Tot: 22s909ms | Train >> Loss: 0.133 | Acc: 98.032% (49016/50000) 98/98 
     [=============================================================>...]  Step: 45ms | Tot: 1s820ms | Test >> Loss: 0.309 | Acc: 91.040% (9104/10000) 20/20 
    ```
    
  - Full
  
  ```python
    
  EPOCH: 1 LR: 0.00125
    
    Epoch: 1
    /content/drive/My Drive/eva-4/assignment-11/models/s11model.py:73: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
      x=F.log_softmax(x)
     [================================================================>]  Step: 154ms | Tot: 22s380ms | Train >> Loss: 1.673 | Acc: 39.958% (19979/50000) 98/98 
     [=============================================================>...]  Step: 42ms | Tot: 1s728ms | Test >> Loss: 1.316 | Acc: 53.360% (5336/10000) 20/20 
    
    EPOCH: 2 LR: 0.0035000000000000005
    
    Epoch: 2
     [================================================================>]  Step: 153ms | Tot: 22s426ms | Train >> Loss: 1.169 | Acc: 58.118% (29059/50000) 98/98 
     [=============================================================>...]  Step: 45ms | Tot: 1s816ms | Test >> Loss: 1.234 | Acc: 57.530% (5753/10000) 20/20 
    
    EPOCH: 3 LR: 0.005750000000000001
    
    Epoch: 3
     [================================================================>]  Step: 151ms | Tot: 22s458ms | Train >> Loss: 0.879 | Acc: 69.098% (34549/50000) 98/98 
     [=============================================================>...]  Step: 44ms | Tot: 1s735ms | Test >> Loss: 0.826 | Acc: 71.600% (7160/10000) 20/20 
    
    EPOCH: 4 LR: 0.008
    
    Epoch: 4
     [================================================================>]  Step: 153ms | Tot: 22s648ms | Train >> Loss: 0.698 | Acc: 76.018% (38009/50000) 98/98 
     [=============================================================>...]  Step: 44ms | Tot: 1s774ms | Test >> Loss: 0.785 | Acc: 74.130% (7413/10000) 20/20 
    
    EPOCH: 5 LR: 0.01025
    
    Epoch: 5
     [================================================================>]  Step: 156ms | Tot: 22s826ms | Train >> Loss: 0.607 | Acc: 79.950% (39975/50000) 98/98 
     [=============================================================>...]  Step: 41ms | Tot: 1s846ms | Test >> Loss: 0.790 | Acc: 73.970% (7397/10000) 20/20 
    
    EPOCH: 6 LR: 0.0125
    
    Epoch: 6
     [================================================================>]  Step: 156ms | Tot: 22s886ms | Train >> Loss: 0.582 | Acc: 81.334% (40667/50000) 98/98 
     [=============================================================>...]  Step: 47ms | Tot: 1s828ms | Test >> Loss: 0.908 | Acc: 69.510% (6951/10000) 20/20 
    
    EPOCH: 7 LR: 0.011875
    
    Epoch: 7
     [================================================================>]  Step: 158ms | Tot: 23s18ms | Train >> Loss: 0.564 | Acc: 82.616% (41308/50000) 98/98 
     [=============================================================>...]  Step: 47ms | Tot: 1s782ms | Test >> Loss: 0.927 | Acc: 68.350% (6835/10000) 20/20 
    
    EPOCH: 8 LR: 0.011250000000000001
    
    Epoch: 8
     [================================================================>]  Step: 156ms | Tot: 23s41ms | Train >> Loss: 0.553 | Acc: 83.126% (41563/50000) 98/98 
     [=============================================================>...]  Step: 44ms | Tot: 1s711ms | Test >> Loss: 0.840 | Acc: 70.490% (7049/10000) 20/20 
    
    EPOCH: 9 LR: 0.010625
    
    Epoch: 9
     [================================================================>]  Step: 155ms | Tot: 22s920ms | Train >> Loss: 0.521 | Acc: 84.378% (42189/50000) 98/98 
     [=============================================================>...]  Step: 46ms | Tot: 1s889ms | Test >> Loss: 0.957 | Acc: 66.870% (6687/10000) 20/20 
    
    EPOCH: 10 LR: 0.01
    
    Epoch: 10
     [================================================================>]  Step: 159ms | Tot: 22s868ms | Train >> Loss: 0.505 | Acc: 84.672% (42336/50000) 98/98 
     [=============================================================>...]  Step: 42ms | Tot: 1s745ms | Test >> Loss: 1.053 | Acc: 64.270% (6427/10000) 20/20 
    
    EPOCH: 11 LR: 0.009375
    
    Epoch: 11
     [================================================================>]  Step: 155ms | Tot: 22s972ms | Train >> Loss: 0.482 | Acc: 85.308% (42654/50000) 98/98 
     [=============================================================>...]  Step: 47ms | Tot: 1s725ms | Test >> Loss: 0.694 | Acc: 77.530% (7753/10000) 20/20 
    
    EPOCH: 12 LR: 0.00875
    
    Epoch: 12
     [================================================================>]  Step: 156ms | Tot: 22s993ms | Train >> Loss: 0.454 | Acc: 86.468% (43234/50000) 98/98 
     [=============================================================>...]  Step: 48ms | Tot: 1s763ms | Test >> Loss: 0.656 | Acc: 79.030% (7903/10000) 20/20 
    
    EPOCH: 13 LR: 0.008125
    
    Epoch: 13
     [================================================================>]  Step: 155ms | Tot: 22s816ms | Train >> Loss: 0.435 | Acc: 87.096% (43548/50000) 98/98 
     [=============================================================>...]  Step: 43ms | Tot: 1s860ms | Test >> Loss: 0.791 | Acc: 73.080% (7308/10000) 20/20 
    
    EPOCH: 14 LR: 0.007500000000000001
    
    Epoch: 14
     [================================================================>]  Step: 156ms | Tot: 23s16ms | Train >> Loss: 0.408 | Acc: 87.928% (43964/50000) 98/98 
     [=============================================================>...]  Step: 43ms | Tot: 1s832ms | Test >> Loss: 1.138 | Acc: 63.560% (6356/10000) 20/20 
    
    EPOCH: 15 LR: 0.006875
    
    Epoch: 15
     [================================================================>]  Step: 161ms | Tot: 22s901ms | Train >> Loss: 0.392 | Acc: 88.448% (44224/50000) 98/98 
     [=============================================================>...]  Step: 43ms | Tot: 1s820ms | Test >> Loss: 0.555 | Acc: 82.380% (8238/10000) 20/20 
    
    EPOCH: 16 LR: 0.0062499999999999995
    
    Epoch: 16
     [================================================================>]  Step: 156ms | Tot: 22s897ms | Train >> Loss: 0.370 | Acc: 89.310% (44655/50000) 98/98 
     [=============================================================>...]  Step: 48ms | Tot: 1s938ms | Test >> Loss: 0.557 | Acc: 81.830% (8183/10000) 20/20 
    
    EPOCH: 17 LR: 0.005624999999999999
    
    Epoch: 17
     [================================================================>]  Step: 154ms | Tot: 22s869ms | Train >> Loss: 0.336 | Acc: 90.516% (45258/50000) 98/98 
     [=============================================================>...]  Step: 47ms | Tot: 1s735ms | Test >> Loss: 0.552 | Acc: 82.510% (8251/10000) 20/20 
    
    EPOCH: 18 LR: 0.005
    
    Epoch: 18
     [================================================================>]  Step: 162ms | Tot: 22s901ms | Train >> Loss: 0.315 | Acc: 91.340% (45670/50000) 98/98 
     [=============================================================>...]  Step: 41ms | Tot: 1s705ms | Test >> Loss: 0.778 | Acc: 74.580% (7458/10000) 20/20 
    
    EPOCH: 19 LR: 0.004375
    
    Epoch: 19
     [================================================================>]  Step: 156ms | Tot: 22s921ms | Train >> Loss: 0.290 | Acc: 92.236% (46118/50000) 98/98 
     [=============================================================>...]  Step: 44ms | Tot: 1s787ms | Test >> Loss: 0.503 | Acc: 84.360% (8436/10000) 20/20 
    
    EPOCH: 20 LR: 0.00375
    
    Epoch: 20
     [================================================================>]  Step: 155ms | Tot: 22s849ms | Train >> Loss: 0.260 | Acc: 93.296% (46648/50000) 98/98 
     [=============================================================>...]  Step: 45ms | Tot: 1s791ms | Test >> Loss: 0.476 | Acc: 84.980% (8498/10000) 20/20 
    
    EPOCH: 21 LR: 0.0031249999999999993
    
    Epoch: 21
     [================================================================>]  Step: 152ms | Tot: 22s989ms | Train >> Loss: 0.229 | Acc: 94.486% (47243/50000) 98/98 
     [=============================================================>...]  Step: 46ms | Tot: 1s795ms | Test >> Loss: 0.519 | Acc: 83.520% (8352/10000) 20/20 
    
    EPOCH: 22 LR: 0.0025000000000000005
    
    Epoch: 22
     [================================================================>]  Step: 153ms | Tot: 22s789ms | Train >> Loss: 0.199 | Acc: 95.626% (47813/50000) 98/98 
     [=============================================================>...]  Step: 46ms | Tot: 1s728ms | Test >> Loss: 0.493 | Acc: 84.440% (8444/10000) 20/20 
    
    EPOCH: 23 LR: 0.001875
    
    Epoch: 23
     [================================================================>]  Step: 156ms | Tot: 22s905ms | Train >> Loss: 0.163 | Acc: 96.956% (48478/50000) 98/98 
     [=============================================================>...]  Step: 49ms | Tot: 1s746ms | Test >> Loss: 0.342 | Acc: 89.460% (8946/10000) 20/20 
    
    EPOCH: 24 LR: 0.0012499999999999994
    
    Epoch: 24
     [================================================================>]  Step: 157ms | Tot: 22s909ms | Train >> Loss: 0.133 | Acc: 98.032% (49016/50000) 98/98 
     [=============================================================>...]  Step: 45ms | Tot: 1s820ms | Test >> Loss: 0.309 | Acc: 91.040% (9104/10000) 20/20 
  ```
  
- CyclicLR plot

  <img src=".\cyclic-lr-plot.JPG" style="zoom:67%;" />

  

- Analysis

  - No of Epochs : 24
  - Batch size : 512
  - Best Test Acc: 91.04%
  - Best Train Acc: 98.03%
  - Custom Model - S11Model
  - Cutout : Yes
  - Albumentations: Yes
  - Hyperparameters
    - Optimizer: SGD
    - Learning rate: 0.0125
    - Momentum: 0.9
