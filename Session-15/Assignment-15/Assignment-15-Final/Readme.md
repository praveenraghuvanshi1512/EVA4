# EVA4 Assignment 15-Final

**Submitted By:** Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)

[Github Directory - Assignment -15-Final](https://github.com/praveenraghuvanshi1512/EVA4/tree/master/Session-15/Assignment-15/Assignment-15-Final)

[Notebook - Main](https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/EVA_4_S15_A_Final_Mask_Depth_V9_Both.ipynb)

[Notebook - NBViewer](https://nbviewer.jupyter.org/github/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/EVA_4_S15_A_Final_Mask_Depth_V9_Both.ipynb)

[Colab Notebook](https://colab.research.google.com/drive/1sPbrH34KqNbhyrQsjhs9o3_rod1tkpzk?usp=sharing)

[Dataset](https://drive.google.com/drive/folders/1qiwLSBbPbEjmp4olW5rLRMM156FttAwb?usp=sharing)

[TOC]



## Introduction

After a span of 4 months EVA-4 Phase-I has come to an end with final capstone project/assignment. We have worked on diverse set of problems during the course in the field of computer vision. The curriculum and methodology of teaching is in accordance with the latest things happening. Diverse platforms used for collaboration has helped a lot in enhancing the knowledge in the field of computer vision. This course has ignited the spark of a student from within after spending a considerable time as a professional and I wish to continue the trajectory of learning that has been laid. Overall, a very satisfactory experience and big thanks to Rohan.

## Problem

The problem is to predict the masks and depthmap from a dataset of images

Sample Images

- Image(Input)

  <img src=".\assets\input.png" alt="Input images" style="zoom:80%;" />

- Mask(Output)

  <img src=".\assets\actual-mask.png" alt="Actual Mask" style="zoom:80%;" />

- Depthmap(Output)

  <img src=".\assets\actual-depth-map.png" alt="Actual - Depth map" style="zoom:80%;" />

Few aspects about the problem

- An end-to-end Machine learning problem
- Data and resource intensive

Some of the complexity involved in this problem is as follows

- Huge dataset: 800k input images
- Multiple input : Background only and Background + Foreground images
- Multiple output : Mask and depth map
- Colab disconnection/memory issues

## Solution

The problem has background and foreground image as base image and mask and depth as output labels.

Preview of actual vs predicted images

<img src=".\assets\actual-predicted-mask.png" alt="Actual Vs Predicted Masks" style="zoom:80%;" />

<img src=".\assets\actual-predicted-depth.png" alt="Actual Vs Predicted Depth map" style="zoom:80%;" />

The solution provided covers all aspects of a Machine learning Cycle as show below and I'll be covering them in detail.

### Data Preparation

#### Dataset

The dataset contains below set of images

- Background(224 x 224 x 3) : These are images of different scenes such as Library and Classroom

- Foreground(96 x 96 x 3) : These are images of objects such as different breed of dogs with transparent background

- Background + Foreground (224 x 224 x 3): These are superimposed images of foreground(dog) over background(library)

- Masks (96 x 96 x 1): These are segmented images of foreground objects

- DenseDepth (224 x 224 x 3): These are 2-D images displaying depth of objects in an image.

  More details could be found at [Assignment 15-A](https://github.com/praveenraghuvanshi1512/EVA4/tree/master/Session-15/Assignment-15/Assignment-15-A)

#### Augmentation

Data augmentation is a technique to generate synthetic data of various types. It helps avoid overfitting issue seen in object detection.

For masks, whatever augmentation we apply on a input images, same needs to be applied to ground truth masks. This is required to preserve the correspondence between base image and mask. I have applied positional augmentation for masks here as they are more relevant to segmentation. Color augmentation is not useful here as ground truth images are single channel, also in colored image, each segmented area represents a class and on applying color augmentation to a image might change class such as from dot to cat. 

The above positional augmentation holds good for depth map as well.

I have used [albumentation](https://github.com/albumentations-team/albumentations) as the augmentation technique. Augmentation used other than normalization and ToTensor is given below.

- **Rotate(-30, 30):** Rotates images by 30 degrees on left and right. This is required as background contains images of library and dog which might be captured at some angles
- **HorizontalFlip:** Flips images by 90 degrees horizontally. This is  required due to different orientation of images

[Augmentation source code](https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/augmentation.py) 

#### Dataset Loading

Loading full dataset(600k) of images at one shot is a challenge and will throw CUDA out of memory exception.

Optimization performed to avoid CUDA OOM issue

- In the beginning, generated full dataset and saved it in google drive. On loading it gave CUDA OOM error
- To overcome above issue, split the dataset into multiple folders with each folder containing 10k images each of background+foreground, mask and depth map with a total of 30k images. This worked but inefficient as it was taking lots of space in google drive
- Another optimization was to create zip files and directly load and create dataset out of that.
- Batch size is very important as it tells how many images to be loaded into memory during one train/test cycle. I have used 16 and it worked quite well.

Dataset was loaded by custom class(ImageDataset) derived from torch Dataset and different arrays are created to hold the information about images. [ImageDataset source code](https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/dataset.py). Directory containing zip files in [google drive](https://drive.google.com/drive/folders/1qiwLSBbPbEjmp4olW5rLRMM156FttAwb?usp=sharing). It has files with names such as output_1.zip, output_2.zip.... output_20.zip. Subfolder in the zip file are as follows.

- bg_fg_1 : 10k background+foreground images
- bg_fg_mask_1 : 10k mask images
- depthMap : 10k depth map images

**Alternative**: I tried to create h5 and binary considering it might reduce the size of dataset, however the size was more. A zip file of 30k images was of 20MB where as h5 and binary it came out to be 200MB, hence it was not used.

### Hyperparameters

- Batch Size(BS) : 16 - As the number of images to be processed is quite high, need to keep this number low in order to avoid CUDA OOM exception.
- Learning Rate(LR) : 0.01 - ReduceLROnPlateau scheduler strategy is used with patience of 2, monitoring validation loss
- Momentum : 0.9
- weight_decay:  1e-5

### Model

- **Model Architecture**

  The problem has a requirement of having an output of similar size as of the input in order to make a pixel wise comparison that would help in predicting both mask and depth map. For both the outputs, we needed an architecture that should generate output of similar size as that input. In case of mask, we are detecting object boundaries and creating a segment/mask. In any DNN, initial layers determines edges and gradients which is required to create a mask/segment in the output image. If we consider a normal DNN such as VGG16, mask won't be generated as last layers predicts the objects such as dog/cat and not the edges. Also, size of the output image shrinks on every convolution if padding is not used. If padding is used computations will be high and still mask won't be determined. 

  In order to overcome above limitation, we needed a network that should generate an image of size similar to input and that can pass of information learned in initial layer such as edges and gradients to the layers in the last. 

  [U-Net](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47) is an architecture for predicting the mask and generates image of similar size. In this architecture, the information in the early layer is passed/concatenated to later layers for predicting the mask. 

  <img src=".\assets\u-net.png" alt="U-Net" style="zoom:80%;" />

  This architecture is based on down/up sample or encode/decode of image. The architecture implemented by my also based on it and adjusted as per this assignment requirement which include multiple input/output, image size, dataset type etc.

  The original architecture did not use batch normalization, after adding it both mask and depth map has improved to a great extent.

  The modified structure is present in summary

  

- **Parameters :** 4,753,252

  When started created a network with huge parameters(57M), the network started with 64 channels and goes till 768 on downsampling and reducing from 768 to 64 on upsampling. This created a network with huge parameters and many layers.

  As the dataset was huge, it will be very time consuming. In order to reduce the parameters, downsampling is done from 64 to 128 and upsampling from 128 to 64. Now the parameters are 4M.

- **Summary**

  ```
  ----------------------------------------------------------------
          Layer (type)               Output Shape         Param #
  ================================================================
              Conv2d-1         [-1, 32, 224, 224]             864
         BatchNorm2d-2         [-1, 32, 224, 224]              64
                ReLU-3         [-1, 32, 224, 224]               0
        ConvBnRelu2d-4         [-1, 32, 224, 224]               0
              Conv2d-5         [-1, 32, 224, 224]           9,216
         BatchNorm2d-6         [-1, 32, 224, 224]              64
                ReLU-7         [-1, 32, 224, 224]               0
        ConvBnRelu2d-8         [-1, 32, 224, 224]               0
             Encoder-9  [[-1, 32, 224, 224], [-1, 32, 112, 112]]               0
             Conv2d-10         [-1, 64, 112, 112]          18,432
        BatchNorm2d-11         [-1, 64, 112, 112]             128
               ReLU-12         [-1, 64, 112, 112]               0
       ConvBnRelu2d-13         [-1, 64, 112, 112]               0
             Conv2d-14         [-1, 64, 112, 112]          36,864
        BatchNorm2d-15         [-1, 64, 112, 112]             128
               ReLU-16         [-1, 64, 112, 112]               0
       ConvBnRelu2d-17         [-1, 64, 112, 112]               0
            Encoder-18  [[-1, 64, 112, 112], [-1, 64, 56, 56]]               0
             Conv2d-19          [-1, 128, 56, 56]          73,728
        BatchNorm2d-20          [-1, 128, 56, 56]             256
               ReLU-21          [-1, 128, 56, 56]               0
       ConvBnRelu2d-22          [-1, 128, 56, 56]               0
             Conv2d-23          [-1, 128, 56, 56]         147,456
        BatchNorm2d-24          [-1, 128, 56, 56]             256
               ReLU-25          [-1, 128, 56, 56]               0
       ConvBnRelu2d-26          [-1, 128, 56, 56]               0
            Encoder-27  [[-1, 128, 56, 56], [-1, 128, 28, 28]]               0
             Conv2d-28          [-1, 256, 28, 28]         294,912
        BatchNorm2d-29          [-1, 256, 28, 28]             512
               ReLU-30          [-1, 256, 28, 28]               0
       ConvBnRelu2d-31          [-1, 256, 28, 28]               0
             Conv2d-32          [-1, 256, 28, 28]         589,824
        BatchNorm2d-33          [-1, 256, 28, 28]             512
               ReLU-34          [-1, 256, 28, 28]               0
       ConvBnRelu2d-35          [-1, 256, 28, 28]               0
            Encoder-36  [[-1, 256, 28, 28], [-1, 256, 14, 14]]               0
             Conv2d-37          [-1, 256, 14, 14]         589,824
        BatchNorm2d-38          [-1, 256, 14, 14]             512
               ReLU-39          [-1, 256, 14, 14]               0
       ConvBnRelu2d-40          [-1, 256, 14, 14]               0
             Conv2d-41          [-1, 256, 14, 14]         589,824
        BatchNorm2d-42          [-1, 256, 14, 14]             512
               ReLU-43          [-1, 256, 14, 14]               0
       ConvBnRelu2d-44          [-1, 256, 14, 14]               0
             Conv2d-45          [-1, 128, 28, 28]         589,824
        BatchNorm2d-46          [-1, 128, 28, 28]             256
               ReLU-47          [-1, 128, 28, 28]               0
       ConvBnRelu2d-48          [-1, 128, 28, 28]               0
             Conv2d-49          [-1, 128, 28, 28]         147,456
        BatchNorm2d-50          [-1, 128, 28, 28]             256
               ReLU-51          [-1, 128, 28, 28]               0
       ConvBnRelu2d-52          [-1, 128, 28, 28]               0
             Conv2d-53          [-1, 128, 28, 28]         147,456
        BatchNorm2d-54          [-1, 128, 28, 28]             256
               ReLU-55          [-1, 128, 28, 28]               0
       ConvBnRelu2d-56          [-1, 128, 28, 28]               0
            Decoder-57          [-1, 128, 28, 28]               0
             Conv2d-58           [-1, 64, 56, 56]         147,456
        BatchNorm2d-59           [-1, 64, 56, 56]             128
               ReLU-60           [-1, 64, 56, 56]               0
       ConvBnRelu2d-61           [-1, 64, 56, 56]               0
             Conv2d-62           [-1, 64, 56, 56]          36,864
        BatchNorm2d-63           [-1, 64, 56, 56]             128
               ReLU-64           [-1, 64, 56, 56]               0
       ConvBnRelu2d-65           [-1, 64, 56, 56]               0
             Conv2d-66           [-1, 64, 56, 56]          36,864
        BatchNorm2d-67           [-1, 64, 56, 56]             128
               ReLU-68           [-1, 64, 56, 56]               0
       ConvBnRelu2d-69           [-1, 64, 56, 56]               0
            Decoder-70           [-1, 64, 56, 56]               0
             Conv2d-71         [-1, 32, 112, 112]          36,864
        BatchNorm2d-72         [-1, 32, 112, 112]              64
               ReLU-73         [-1, 32, 112, 112]               0
       ConvBnRelu2d-74         [-1, 32, 112, 112]               0
             Conv2d-75         [-1, 32, 112, 112]           9,216
        BatchNorm2d-76         [-1, 32, 112, 112]              64
               ReLU-77         [-1, 32, 112, 112]               0
       ConvBnRelu2d-78         [-1, 32, 112, 112]               0
             Conv2d-79         [-1, 32, 112, 112]           9,216
        BatchNorm2d-80         [-1, 32, 112, 112]              64
               ReLU-81         [-1, 32, 112, 112]               0
       ConvBnRelu2d-82         [-1, 32, 112, 112]               0
            Decoder-83         [-1, 32, 112, 112]               0
             Conv2d-84         [-1, 32, 224, 224]          18,432
        BatchNorm2d-85         [-1, 32, 224, 224]              64
               ReLU-86         [-1, 32, 224, 224]               0
       ConvBnRelu2d-87         [-1, 32, 224, 224]               0
             Conv2d-88         [-1, 32, 224, 224]           9,216
        BatchNorm2d-89         [-1, 32, 224, 224]              64
               ReLU-90         [-1, 32, 224, 224]               0
       ConvBnRelu2d-91         [-1, 32, 224, 224]               0
             Conv2d-92         [-1, 32, 224, 224]           9,216
        BatchNorm2d-93         [-1, 32, 224, 224]              64
               ReLU-94         [-1, 32, 224, 224]               0
       ConvBnRelu2d-95         [-1, 32, 224, 224]               0
            Decoder-96         [-1, 32, 224, 224]               0
             Conv2d-97          [-1, 1, 224, 224]              33
             Conv2d-98          [-1, 128, 28, 28]         589,824
        BatchNorm2d-99          [-1, 128, 28, 28]             256
              ReLU-100          [-1, 128, 28, 28]               0
      ConvBnRelu2d-101          [-1, 128, 28, 28]               0
            Conv2d-102          [-1, 128, 28, 28]         147,456
       BatchNorm2d-103          [-1, 128, 28, 28]             256
              ReLU-104          [-1, 128, 28, 28]               0
      ConvBnRelu2d-105          [-1, 128, 28, 28]               0
            Conv2d-106          [-1, 128, 28, 28]         147,456
       BatchNorm2d-107          [-1, 128, 28, 28]             256
              ReLU-108          [-1, 128, 28, 28]               0
      ConvBnRelu2d-109          [-1, 128, 28, 28]               0
           Decoder-110          [-1, 128, 28, 28]               0
            Conv2d-111           [-1, 64, 56, 56]         147,456
       BatchNorm2d-112           [-1, 64, 56, 56]             128
              ReLU-113           [-1, 64, 56, 56]               0
      ConvBnRelu2d-114           [-1, 64, 56, 56]               0
            Conv2d-115           [-1, 64, 56, 56]          36,864
       BatchNorm2d-116           [-1, 64, 56, 56]             128
              ReLU-117           [-1, 64, 56, 56]               0
      ConvBnRelu2d-118           [-1, 64, 56, 56]               0
            Conv2d-119           [-1, 64, 56, 56]          36,864
       BatchNorm2d-120           [-1, 64, 56, 56]             128
              ReLU-121           [-1, 64, 56, 56]               0
      ConvBnRelu2d-122           [-1, 64, 56, 56]               0
           Decoder-123           [-1, 64, 56, 56]               0
            Conv2d-124         [-1, 32, 112, 112]          36,864
       BatchNorm2d-125         [-1, 32, 112, 112]              64
              ReLU-126         [-1, 32, 112, 112]               0
      ConvBnRelu2d-127         [-1, 32, 112, 112]               0
            Conv2d-128         [-1, 32, 112, 112]           9,216
       BatchNorm2d-129         [-1, 32, 112, 112]              64
              ReLU-130         [-1, 32, 112, 112]               0
      ConvBnRelu2d-131         [-1, 32, 112, 112]               0
            Conv2d-132         [-1, 32, 112, 112]           9,216
       BatchNorm2d-133         [-1, 32, 112, 112]              64
              ReLU-134         [-1, 32, 112, 112]               0
      ConvBnRelu2d-135         [-1, 32, 112, 112]               0
           Decoder-136         [-1, 32, 112, 112]               0
            Conv2d-137         [-1, 32, 224, 224]          18,432
       BatchNorm2d-138         [-1, 32, 224, 224]              64
              ReLU-139         [-1, 32, 224, 224]               0
      ConvBnRelu2d-140         [-1, 32, 224, 224]               0
            Conv2d-141         [-1, 32, 224, 224]           9,216
       BatchNorm2d-142         [-1, 32, 224, 224]              64
              ReLU-143         [-1, 32, 224, 224]               0
      ConvBnRelu2d-144         [-1, 32, 224, 224]               0
            Conv2d-145         [-1, 32, 224, 224]           9,216
       BatchNorm2d-146         [-1, 32, 224, 224]              64
              ReLU-147         [-1, 32, 224, 224]               0
      ConvBnRelu2d-148         [-1, 32, 224, 224]               0
           Decoder-149         [-1, 32, 224, 224]               0
            Conv2d-150          [-1, 3, 224, 224]              99
  ================================================================
  Total params: 4,753,252
  Trainable params: 4,753,252
  Non-trainable params: 0
  ----------------------------------------------------------------
  Input size (MB): 0.57
  Forward/backward pass size (MB): 6530073.81
  Params size (MB): 18.13
  Estimated Total Size (MB): 6530092.52
  ----------------------------------------------------------------
  ```

  

- [Source code](https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/models/custommodel.py)

### Loss function and Optimizers

#### Loss Function : 

I have used Binary Cross Entropy(BCE) loss for this assignment after considering below losses

- BCE - The loss here was less compared to all the below losses.
- Dice - The loss reported was almost double compared to BCE 
- Kornea - This didn't work well and gave black images for masks

The problem is to predict two images(mask and depthmap). A single loss won't be sufficient as loss from pixel comparison from a mask will be different than that of depthmap. During training, it was observed that mask is predicted quite well on large dataset whereas depth was not predicted quite well. So, added more weightage of depth loss compared to mask for calculating average loss and it improved the depthmap. 

[Source code](https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-Final/functions.py)

#### Optimizer : 

The original U-Net paper used SGD and I too have stick to the same.

### Train/Test Model

- Epochs
  - First Zip file : 40
  - Second : 20
  - Third to Twenty: 10
- Evaluation criteria
  - Loss - Training and test loss is measured and a plot was created to evaluate the performance of model
  - Visual diff of actual and predicted images

### Result

- Training loss ~ 0.3
- Validation loss ~ 0.3
- Refer actual and predicted images in the solution section

### Save and Load Models

- Models were trained and best model was saved along with loss and epoch
- Previously saved models were loaded and trained further on next set of zip file
- It has helped a lot in overcoming uncertain issues in colab

### Plots

- A plot was generated for each of the trained zip file having Train/Test loss

  <img src=".\assets\train-test-loss-curve.png" alt="Train vs Test loss" style="zoom:80%;" />

### Optimizations performed

- Resize images to 224x224 square
- Remove unnecessary print statements
- Remove directory/file iterating statements used for print
- printing images at certain interval and not on every epoch
- Removed unused code
- Model size : Parameters
- Model Checkpoint
- Split the dataset
- Batch Normalization

### Code Modularization

- The Code has been modularized as below
  - **EVA-4-S11-Praveen-Raghuvanshi-Main-91-04.ipynb** : It contains main workflow
  - **functions.py**: It contains all the functions related to setup, train, load dataset etc.
  - **models/custommodel.py** : It contains Custom Net Model
  - **augmentation.py**: It contains image augmentations such as albumentations
  - Code structured to have similar items together such as constants, import statements etc.

### Challenges

- Tensorboard didn't work as it was consuming lot of time
- Multiple input/output
- Initially mask was not getting predicted however it was predicted in case of network for detecting masks only
- Huge dataset

### Helpers

- autotime : Prints cell execution time

### Further Improvements

- Different optimizers
- Image augmentation
- Optimizations (Memory management, Performance and channel/layers)
- Train on large images such as 1024 x 1024 where ground truth for depth map will be better

## Conclusion

This assignment is a great learning exercise. I have been exposed to lot of new things such as handling multiple input/ouput/loss within a single network. An great use of ResNet like structure such as U-Net. Handling huge dataset and training for 600k images which seems to be impossible in the beginning. Definitely there is a scope of improvement in the solution provided.

## References

- [Understanding Semantic Segmentation with UNET](https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47)

- [How To Save and Load Model In PyTorch With A Complete Example](https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee)

- [Image Segmentation using U-Net](https://www.youtube.com/watch?reload=9&v=azM57JuQpQI&feature=emb_rel_pause)

- [Pytorch Mono Depth](https://github.com/simonmeister/pytorch-mono-depth)

- [Tensorboard](http://www.programmersought.com/article/8842149639/)

  