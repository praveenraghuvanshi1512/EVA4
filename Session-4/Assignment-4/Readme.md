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
- Validation Accuracy: 
- No of parameters: 
- No of Epochs
- Model
- Logs
- Hyperparameters
  - 
- Activation Function: 
- Optimizer: 
- No of layers
- No of blocks
- Learning Rate: 

#### Analysis

- Dataset: MNIST

- Image size: 28 x 28 x 1

- Train data: 60000, Test data: 10000

- Classes (10) : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 

- Images are of same size, centered and size normalized

- MNIST is a labeled dataset and falls under supervised learning

  

| S.No | Iteration       | # Parameters | Val Acc (Best)     | # layers | Time(min) | Model Changes                                                | Remark                                   |
| ---- | --------------- | ------------ | ------------------ | -------- | --------- | ------------------------------------------------------------ | ---------------------------------------- |
| 1    | Base            | 1,172, 410   | 81.88 (50th Epoch) | 26       | 6.2       | No Change                                                    | N/A                                      |
| 2    | Solution(Final) | 84,277       | 83.98(Best)        |          | 47        | Refer Iteration# 15                                          | Best acccuracy                           |
| 3    | First           | 604,213      | 100                | 26       | 10        | Conv2D --> SeparableConv2D                                   | Parameters reduced, Acc is constant 1.00 |
| 4    | Second          | 609,973      | 82.18              | 32       | 12.8      | Batch normalization (SepConv -> BN -> ReLU -> SepConv -> BN -> ReLU -> MP) | Parameters increased, Accuracy is normal |
| 5    | Third           | 93,109       | 78.80              | 30       | 12.9      | Remove Dense Layer D1(393,728) and D2(131,328)               | Parameters is under 100,000, Acc reduced |
| 6    | Fourth          | --           | 80.50              | 29       | 12.9      | Remove Dropout from last layer                               | Acc improved                             |
| 7    | Fifth           | --           | 80.92              | 29       | 12.9      | Reduce Dropout from 0.25 to 0.1                              | Acc improved                             |
| 8    | Sixth           | --           | 78.98              | 27       | 12.9      | Remove all Dropout                                           | Acc reduced                              |
| 9    | Seventh         | --           | 69.68              | 27       | 12.9      | Brought Dropout back. Fifth                                  | Acc reduced                              |
| 10   | Eighth          | --           | 79.91              | 29       | 20        | Increase Batch size (128 --> 256)                            | Acc improved                             |
| 11   | Ninth           | --           | 77.93              | --       | 19        | Increased batch size (128 --> 256 --> 512)                   | Acc reduced                              |
| 12   | Tenth           | --           | 81.40              | --       | 28        | Decrease batch size (128 --> 64)                             | Acc improved                             |
| 13   | Eleventh        | --           | 82.13              | --       | 41        | Decrease batch size (128 --> 64 --> 32)                      | Acc improved and crossed base (81.88%)   |
| 14   | Twelveth        | 84,277       | 82.44              |          | 38        | Last Dense and Flatten layer replaced with GAP               | Parameters reduced, Acc improved         |
| 15   | Thirteenth      | --           | 69.48              |          | 43        | Image Augmentation (horizontal and vertical flip, rotation)  | Acc reduced drastically to 69.48         |
| 16   | Fourteenth      | --           |                    |          |           |                                                              |                                          |
| 17   | Fifteenth       | --           | 83.11              |          | 30        | Image Augmentation(horizontal flip, rotation range, slide(height and width)) | Acc improved with best accuracy          |



### Iterations

##### First

- A basic network involving only convolutions till the RF is equal to size of image
- 