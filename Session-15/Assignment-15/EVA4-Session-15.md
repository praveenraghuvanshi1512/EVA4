# 	 EVA15 - Session 15: 03-May-2020

#### Video  : https://youtu.be/zVQ6eSYXBq4

[![EVA-4 Session 15](http://img.youtube.com/vi/zVQ6eSYXBq4/0.jpg)](https://youtu.be/zVQ6eSYXBq4)

### Links

- [Session-15 Pdf](S15.pdf)

- [Video](https://youtu.be/zVQ6eSYXBq4)

#### Assignment

- Assignment description is already shared in the last assignment:
  - Given an image with foreground objects and background image, predict the depth map as well as a mask for the foreground object. 
- It is an open problem and you can solve it any way you want. 
- Let's look at how it can be approached through some examples. 
- Assignment 14 (15A )was given to start preparing you for assignment 15th. 14th (15A) automatically becomes critical to work on the 15th. 
- **The 15th assignment is NOT a group assignment**. You are supposed to submit it along. 
- What happens when you copy? Well, WHAT happens when we copy? WHO knows!
- This assignment is worth 10,000 points. 
- Assignment 15th is THE qualifying assignment. 

 

**Evaluation:**

- A very heavy score on documenting what you have done. If you submit just files without documenting what you have (gone through to do/) done, then even if you get perfect results, you won't get more than 40%. 
- A heavy score for the use of Modular code (especially if it is from the package you have written)
- A heavy score for your data management skills
  - have you thought about changing data format, channels, etc
  - have you thought and utilized the fact that you can train on smaller images first and then move to large resolution ones
  - how creative you have been, for example, what all loss functions have you tried, have you tried to only solve for 1 problem before solving for both, etc
  - how creative your DNN is, how many params you have (a DNN without Param count mentioned will not get any evaluation points from us as we won't know did it actually help)
  - how creative you have been while using data augmentation, and why did you pick those augmentations
  - colab gives you less than 9-10 hours, then how did you manage creatively compute for days
  - have you done any analysis on how much time each block (dnn, data prep, loss calc, etc) takes (basic python "timeit" at least!)
  - how have you presented your results? 
    - we are now talking about the accuracy of the depth map and area of the foreground. How would you present your accuracy now?
    - have you just thrown some logs/numbers on the screen or have actually presented the visual results as well. 

 

Assignment 15 will test whether you have come out of the "novice" shell or not. Anything which looks too primitive or "not well thought off", will attract a negative penalty. 

 

Once done submit your assignment 15 GitHub link. We will **ONLY** read your readme file, and unless your readme file directs us to read any other file/notebook we will not open anything. 

Evaluators are lazy, to make sure that you can keep them excited when they are reading your readme file, and think how to force them to look at your code. 

- Video notes

  1. Select a background (bg)

  2. Select some foreground (fg) : It can be cropped

  3. Place foreground randomly on background (bg_fg) - Main image

  4. Take bg_fg and send it to 3d monocular network

  5. Output of 4 will be Depth map : Most of the people are doing depth map on CMap(Color map), try to visualize in greyscale

  6. If we take CMap of greyscale, we'll get a much better perspective of what is happening

  7. Problem to be solved

     1. Given an image (bg_fg) and background image(bg)
     2. Network must take both the images
     3. Process them and finally spit out
        1. 3d Depth map
        2. Mask - Foreground(fg) objects

     <img src=".\assets\depth-estimation-assignment.png" alt="Session-15 Problem statement" style="zoom:50%;" />

     ​		<img src=".\assets\depth-mask-1.png" alt="Depth Mask Foreground" style="zoom:50%;" />		

      

     8. Solution hints

        1. Don't try to solve full problem. Solve it step by step

        2. First is to predict backgrounds, predicting foregrounds basically

        3. Can i create a mask directly?

        4. Load dataset from different directories such as bg, bg_fg, mask

        5. Scale transformations on - Resize, ToTensor

        6. Calculate mean and std

        7. Apply train transformation - Resize, ColorJitter, ToTensor

        8. Load train dataset

        9. Print all three dataset shapes(bg, bg_fg, mask)

        10. Create train Dataloader with dataset

        11. Calculate grid_tensor, grid_image

        12. Display background images - 16

        13. Create a simple neural network

        14. Run model through above network

        15. Save models

        16. Train model

        17. Check output after first epoch and look at the mask generated

        18. Train further for better mask

        19. Look at the loss 

        20. Look at the mask and try to remove patches 

        21. The problem is trying to figure out what is the loss

        22. Loss is dependent on criteria which is BCEWithLogitsLoss

        23. Read about all the losses and thats the target to lower the loss

        24. Another thought is convert images into Edge

        25. If our mask has to be accurate, don't you think the edge map of my mask must be equal to ground truth edge map

        26. If we calculate edge of our mask with the ground truth edge map, they must match and that is going to give us perfect bounding box

        27. Loss has to be calculated per pixel

        28. We can use L1, L2, SSIM, BCE ...

        29. Identifying strategy for loss is the main assignment

        30. After every epoch, we'll print actual image, depth map, mask, fg and bg

        31. With above, colab RAM will increase and it'll crash

        32. Keep logs very simple

        33. Move to Tensorboard

        34. See the results 

            <img src=".\assets\a-15-sample-result.png" alt="A-15-Sample Result" style="zoom:80%;" />

            <img src=".\assets\a-15-sample-result-2.png" alt="a-15-sample-result-2" style="zoom:80%;" />

            

        35. Sometimes foreground(fg) is pasted on background(bg) and sometimes not. Its upto us

        36. Ground truth

            <img src=".\assets\a-15-sample-result-ground-truth.png" alt="a-15-sample-result-ground-truth" style="zoom:80%;" />

        37. **Prediction**

            ![Result - Predicted](.\assets\a-15-sample-result-prediction.png)

        38. Trained for 20 epochs only

        39. **Mask  Ground Truth**

            <img src=".\assets\a-15-sample-result-mask-ground-truth.png" alt="Mask - Ground truth" style="zoom:80%;" />

        40. **Mask - Predicted**

            <img src=".\assets\a-15-sample-result-mask-predicted.png" alt="Mask - Predicted" style="zoom:80%;" />

        41. 

        42. Explaining approach to the problem is most important

        43. PNG or JPG

        44. JPG(26) < PNG(83)

        45. The images we are dealing with have textures, and we don't use PNG's for textures, PNG's are lossless  compression

        46. We'll reduce the size of image and they are in JPG format. 

        47. Anyway they have to be converted to binary format when Pytorch is going to read it.

        48. Mask has 1 channel, what is the point of storing them in 4-channel PNG. This will lead to 4-times RAM usage

        49. Predict depth in 1-channel only

        50. Start with low resolution images first and move on to high resolution afterwards

        51. For depth, what kind of augmentation should be used

        52. Checkpoint your models

        53. Google chrome plugin for keeping colab active or any script 

        54. Timeit

        55. 

  #### My approach

  - Get fg, bg_fg images
  - Run model
  - Create Depth map
  - 

#### Notes 

- ReLU is fast and has simple differentiability

- Padding is used when important aspects are present on the edges of an image.

- 

- Depth Estimation

- <img src=".\assets\depth-estimation-supervised.png" alt="Depth Estimation Supervised" style="zoom:80%;" />

- Baseline, target

- <img src=".\assets\depth-estimation-unsupervised.png" alt="image-20200512193541705" style="zoom:80%;" />

- Input image

  ![Input](.\assets\depth-estimation-input.png)

- Baseline

  <img src=".\assets\depth-estimation-baseline.png" alt="Baseline" style="zoom:80%;" />

- New method

  <img src=".\assets\depth-estimation-new.png" alt="New Method" style="zoom:80%;" />

  <img src=".\assets\depth-estimation-model.png" alt="image-20200512200445605" style="zoom:80%;" />

- VGG and Resnet50

  <img src=".\assets\depth-estimation-architecture.png" alt="image-20200512200759393" style="zoom:80%;" />

  <img src=".\assets\depth-estimation-conclusion.png" alt="image-20200512201146040" style="zoom:80%;" />

- https://github.com/satyajitghana/TSAI-DeepVision-EVA4.0/tree/master/14_RCNN

- **U-Net**

  <img src=".\assets\mask-u-net.png" alt="Mask - U-Net" style="zoom:80%;" />

- fdsaf

#### TODO

- Prepare dataset for 10000 images
- Mean and std 
- Use Tenserboard in colab
  - Visualize model structure
  - https://hackernoon.com/how-to-structure-a-pytorch-ml-project-with-google-colab-and-tensorboard-7ram3agi
- Save model - Checkpoint
  - https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
- Generalize the network
- Remove images and print at the end
- Add sigmoid to the Mask last convolution
- Optimize memory
- TimeIt 
- Train/Test split
- code for test
- ML Pipeline



#### Analysis

- We used Resnet34 with pretrained weights but it threw an error
  - Size mismatch : Target: [9, 1, 224, 224] and predicted : [9, 3, 1521(13 x 13 x 9)]
- As the task was to identify masks, we needed a network that can preserve edges and gradients predicted in the initial layers and use it in later layers for getting better information about the edges, U-net was the closest architecture to achieve same. 
- Image size of 224 was too big leading to huge dataset size for 400K images
- We tried to reduce the size to 64 as main focus was to have mask and depth map.
- We reduced size from ... to ...
- Add BN and ReLU

##### Video

- 


#### Questions

- Monocular vs stereo image depth estimation

#### Further Study

- 

### References

- Concepts
  - Convolution
    - https://towardsdatascience.com/convolutional-neural-networks-cnns-a-practical-perspective-c7b3b2091aa8
    - https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1
  - Depth Estimation
    - Tutorial
      - https://www.youtube.com/watch?v=jI1Qf7zMeIs
      - https://www.youtube.com/watch?v=HWu39YkGKvI
    - Code
      - https://github.com/simonmeister/pytorch-mono-depth
      - [Notebook - Pytorch : Dhruv Jawalkar](https://github.com/DhruvJawalkar/Depth-Map-Prediction-from-a-Single-Image-using-a-Multi-Scale-Deep-Network/blob/master/depth-prediction.ipynb)
      - https://github.com/gsurma/mono_depth_estimator
      - [Notebook] https://github.com/gsurma/mono_depth_estimator/blob/master/Pix2Pix-DepthEstimation.ipynb
  - Segmentation
    - U-Net : https://www.youtube.com/watch?reload=9&v=azM57JuQpQI&feature=emb_rel_pause
      - https://github.com/usuyama/pytorch-unet
      - https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
      - https://tuatini.me/practical-image-segmentation-with-unet/
      - https://github.com/ugent-korea/pytorch-unet-segmentation
      - https://www.youtube.com/watch?v=uiE56h5LyXc
      - https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6
      - TO READ
        - https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/
        - https://towardsdatascience.com/medical-image-segmentation-part-1-unet-convolutional-networks-with-interactive-code-70f0f17f46c6
        - https://towardsdatascience.com/depth-estimation-on-camera-images-using-densenets-ac454caa893
        - https://mxnet.apache.org/versions/1.2.1/tutorials/python/data_augmentation_with_masks.html
        - https://www.slideshare.net/NaverEngineering/depth-estimation-do-we-need-to-throw-old-things-away
        - IMP [file:///C:/Users/praghuvanshi/Downloads/sensors-19-01795.pdf](file:///C:/Users/praghuvanshi/Downloads/sensors-19-01795.pdf)
- Visualization
  - Tensorboard
    - http://www.programmersought.com/article/8842149639/



## Optimizations

- Resize images to 224x224 square

- Remove unnecessary print statements

- Remove directory/file iterating statements used for print

- printing images at no_of_epochs/10 

- Removed unused code

- Model size 

- callbacks

- Model Checkpoint

- Early stopping

- Split the dataset

- BN

  



### Current Issues

- Running model with two outputs is not working

  - Mask and depth map 
  - Mask : 1 channel
  - Depth : 3 Channel
  - Mask: greyscale
  - Depth: Colored
  - How are we merging and getting output from two layers

- How to use dataset?

- Workaround

  - Use sigmoid for mask
  - Train on 32 size
  - Change loss

- Background images 100 same or different?

- Input : bg, bg_fg

- Output: mask and depth

- Dataset:

  - bg - 100
  - bg_fg - 500(3) : 3, 224, 224
  - masks - 500(1) : 1, 224, 224
  - depths - 500(3) : 3, 224, 224

- Single

  - Mask: 
    - <img src=".\assets\mask-actual.png" alt="Mask Actual" style="zoom:80%;" />
    - <img src=".\assets\mask-predicted-1.png" alt="Mask Predicted" style="zoom:80%;" />
    - <img src=".\assets\mask-predicted-50.png" alt="Mask Predicted - 50 Epoch - Custom" style="zoom:80%;" />
  - Depth
    - <img src=".\assets\depth_actual.png" alt="Actual Depth" style="zoom:80%;" />
    - <img src=".\assets\depth_predicted_1.png" alt="Predicted Depth" style="zoom:80%;" />
    - loss = -32
    - Time: 9min /epoch
    - BS:32

- Both

  - Custom:

    - Mask

      <img src=".\assets\mask-predicted-1-custom-both.png" alt="Both Custom Mask - 1" style="zoom:80%;" />

    - Depth

      <img src=".\assets\depth_predicted_1-custom-both.png" alt="Depth - Custom - 1" style="zoom:80%;" />

  - Time / epoch: 22 mins

  - Colab crashing with OOM

    <img src=".\assets\image-resize-code.png" alt="Image - Resize code" style="zoom:80%;" />

### Misc

- Torchvision dataset

## Results

| S.No | #images/GT               | #Channels               | Model                                                        | #Params    | Loss Fn                | Optimizer | #Epochs | Observed loss | Output | Remarks                                                      |
| ---- | ------------------------ | ----------------------- | ------------------------------------------------------------ | ---------- | ---------------------- | --------- | ------- | ------------- | ------ | ------------------------------------------------------------ |
| 1    | 500(1500)                | in - 3, mask-1, depth-3 | [UNet based- korea](https://raw.githubusercontent.com/ugent-korea/pytorch-unet-segmentation/master/src/simple_model.py) | 14,348,004 | BCEWithLogitsLoss      | SGD       | 100     | 0.72          | Both   | Both predicted. Mask predicted at E-70                       |
| 2    | 10k(30k)                 | in - 3, mask-1, depth-3 | [UNet based- korea](https://raw.githubusercontent.com/ugent-korea/pytorch-unet-segmentation/master/src/simple_model.py) | 14,348,004 | BCEWithLogitsLoss      | SGD       | 100     |               | Both   | 1st epoch took lot of time                                   |
| 3    | 500(1500)                | in - 3, mask-1, depth-3 | [UNet based- korea](https://raw.githubusercontent.com/ugent-korea/pytorch-unet-segmentation/master/src/simple_model.py) - Sigmoid at mask | 14,348,004 | BCEWithLogitsLoss      | SGD       | 300     | 1.20          | depth  | Adding sigmoid worsened  output                              |
| 4    | **500(1500)**            | in - 3, mask-1, depth-3 | [UNet based- korea](https://raw.githubusercontent.com/ugent-korea/pytorch-unet-segmentation/master/src/simple_model.py) | 14,348,004 | BCEWithLogitsLoss only | SGD       | 200     | **0.60**      | Both   | Dice loss is bad, BCE only is good                           |
| 5    | 500(1500)                | in - 3, mask-1, depth-3 | [UNet -korea-advance](https://raw.githubusercontent.com/ugent-korea/pytorch-unet-segmentation/master/src/advanced_model.py) | 57,378,244 | BCEWithLogitsLoss only | SGD       | 200     | 0.55          | Depth  | Large no of parameters,  Good mask and depth in 200 epochs   |
| 6    | 500(1500)                | in - 3, mask-1, depth-3 | [U-Net](https://raw.githubusercontent.com/EKami/carvana-challenge/original_unet/src/nn/unet.py) + BN | 57,188,044 | BCEWithLogitsLoss only | SGD       | 200     | 0.39          |        | BN reduced loss to a great extent. Much deeper network with huge parameters 57M |
| 7    | 10k(30k)                 | in - 3, mask-1, depth-3 | [U-Net](https://raw.githubusercontent.com/EKami/carvana-challenge/original_unet/src/nn/unet.py) + BN | 57,188,044 | BCEWithLogitsLoss only | SGD       | 5       | 0.30          | Both   | Mask start to appear in early epoch. Depth not good          |
| 8    | 10k(**40k**) - Bg images | in - 3, mask-1, depth-3 | [U-Net](https://raw.githubusercontent.com/EKami/carvana-challenge/original_unet/src/nn/unet.py) + BN | 57,188,044 | BCEWithLogitsLoss only | SGD       | 5       | 0.30          | Both   | Mask start to appear in early epoch. Depth not good          |
|      |                          |                         |                                                              |            |                        |           |         |               |        |                                                              |

