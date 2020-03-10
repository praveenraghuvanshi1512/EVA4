# EVA4 Assignment 8 - Praveen Raghuvanshi



- [Github link](https://github.com/praveenraghuvanshi1512/EVA4/blob/Session-8/Session-8/Assignment-8/EVA_4_S8_Praveen_Raghuvanshi_Main.ipynb) 

- [Colab link](https://colab.research.google.com/drive/1RqhZW2A20G3p6mzoTtAXnW0AViS5oByd?authuser=1#scrollTo=EHGIiWEF-RMP)

- Model: Resnet18

- No of parameters: 11,173,962

- No of Epochs : 30

- Best Test Acc: 86.560%

- Epoch 8 Accuracy

  - Train : 86.164%
  - Test : 86.560%

- Code Modularity

  - EVA_4_S8_Praveen_Raghuvanshi_Main.ipynb : It contains main workflow
  - S8_functions.py: It contains all the functions related to setup, train, load dataset etc.
  - model.py : It contains Resnet model
  - Utils.py : It contains helper functions such as progress bar

- Logs

  - Best accuracy

    ```python
    Epoch: 8
     [================================================================>]  Step: 25ms | Tot: 5m36s | Train >> Loss: 0.400 | Acc: 86.164% (43082/50000) 12500/12500 
     [================================================================>]  Step: 20ms | Tot: 47s395ms | Test >> Loss: 0.408 | Acc: 86.560% (8656/10000) 2500/2500 
    
    ```

  - Full

    ```python
    Epoch: 0
     [================================================================>]  Step: 26ms | Tot: 5m43s | Train >> Loss: 1.567 | Acc: 43.442% (21721/50000) 12500/12500 
     [================================================================>]  Step: 15ms | Tot: 48s33ms | Test >> Loss: 1.093 | Acc: 61.430% (6143/10000) 2500/2500 
    
    Epoch: 1
     [================================================================>]  Step: 26ms | Tot: 5m34s | Train >> Loss: 0.995 | Acc: 64.882% (32441/50000) 12500/12500 
     [================================================================>]  Step: 15ms | Tot: 47s357ms | Test >> Loss: 0.807 | Acc: 73.140% (7314/10000) 2500/2500 
    
    Epoch: 2
     [================================================================>]  Step: 22ms | Tot: 5m35s | Train >> Loss: 0.780 | Acc: 72.752% (36376/50000) 12500/12500 
     [================================================================>]  Step: 22ms | Tot: 46s891ms | Test >> Loss: 0.691 | Acc: 76.680% (7668/10000) 2500/2500 
    
    Epoch: 3
     [================================================================>]  Step: 23ms | Tot: 5m38s | Train >> Loss: 0.659 | Acc: 77.024% (38512/50000) 12500/12500 
     [================================================================>]  Step: 16ms | Tot: 47s476ms | Test >> Loss: 0.622 | Acc: 79.220% (7922/10000) 2500/2500 
    
    Epoch: 4
     [================================================================>]  Step: 29ms | Tot: 5m38s | Train >> Loss: 0.583 | Acc: 79.728% (39864/50000) 12500/12500 
     [================================================================>]  Step: 19ms | Tot: 47s864ms | Test >> Loss: 0.538 | Acc: 82.180% (8218/10000) 2500/2500 
    
    Epoch: 5
     [================================================================>]  Step: 23ms | Tot: 5m42s | Train >> Loss: 0.517 | Acc: 82.250% (41125/50000) 12500/12500 
     [================================================================>]  Step: 19ms | Tot: 48s702ms | Test >> Loss: 0.505 | Acc: 82.780% (8278/10000) 2500/2500 
    
    Epoch: 6
     [================================================================>]  Step: 25ms | Tot: 5m37s | Train >> Loss: 0.472 | Acc: 83.850% (41925/50000) 12500/12500 
     [================================================================>]  Step: 13ms | Tot: 46s879ms | Test >> Loss: 0.442 | Acc: 84.630% (8463/10000) 2500/2500 
    
    Epoch: 7
     [================================================================>]  Step: 25ms | Tot: 5m37s | Train >> Loss: 0.433 | Acc: 85.072% (42536/50000) 12500/12500 
     [================================================================>]  Step: 17ms | Tot: 47s651ms | Test >> Loss: 0.447 | Acc: 84.890% (8489/10000) 2500/2500 
    
    Epoch: 8
     [================================================================>]  Step: 25ms | Tot: 5m36s | Train >> Loss: 0.400 | Acc: 86.164% (43082/50000) 12500/12500 
     [================================================================>]  Step: 20ms | Tot: 47s395ms | Test >> Loss: 0.408 | Acc: 86.560% (8656/10000) 2500/2500 
    
    Epoch: 9
     [================================================================>]  Step: 23ms | Tot: 5m33s | Train >> Loss: 0.373 | Acc: 87.184% (43592/50000) 12500/12500 
     [================================================================>]  Step: 17ms | Tot: 47s933ms | Test >> Loss: 0.425 | Acc: 85.560% (8556/10000) 2500/2500
    ```

- Analysis

  - No of Epochs : 9
  - Best Train Acc: 87.184%
  - Best Test Acc: 86.560%
  - Epoch 8 Acccuracy
    - Train : 86.164%
    - Test : 86.560%
  - Overfitting
    - Calculation: (100 - 86.560) + 86.164 = 99.604 --> No Overfitting
    - Difference : 86.164 - 86.560 = -0.396 --> Less -> No overfitting

  

