# EVA4 - Session 2 : 22-Jan-2020

#### Video  : https://youtu.be/pQCGtqDg8nw

[![EVA Session 2](http://img.youtube.com/vi/pQCGtqDg8nw/0.jpg)](https://youtu.be/pQCGtqDg8nw)

### Links

- [PDF](EVA4-Session-2.pdf)
- [Video](https://youtu.be/pQCGtqDg8nw)

#### Notes

- In the below statement, first layer has 32/64 kernels and we know that initial layers are used to find edges and gradients. Does these kernels comprise only edges/gradients or it can be anything such as textures/parts of objects and will be updated during backpropagation

  *We would need a set of edges and gradients to be detected to be able to represent the whole image. Through experiments, we have learned that we should use around 32 or 64 kernels in the first layer, increasing the number of kernels slowly. Let's us assume we add 32 kernels in the first layer, 64 in second, 128 in thrid and so on.* 

- There is no right kernel for a particular dataset. 

- There may be a kernel for an object in different lighting condition of a particular size at a particular location

- A Kernel can't be generalized.

- Just to extract a vertical line, we could have lot of kernels.fdsaf

- DNN filter out things not required. For e.g if we need to focus on a dog with background, DNN needs to filter out the background.

- We have 4 steps(Edges and gradients, textures and patterns, parts of objects and objects). At every step, we filter out undesired things.

- Are we loosing/filtering the information? We are filtering the information. 

- First five sessions, size of object is same as size of image.

  <img src=".\assets\frnot-background-filtering.png" alt="Front-Background Filtering" style="zoom:80%;" />

- Local Receptive field is always equal to size of kernel

- We need to know the Global Receptive field in order to add layers for identifying full object.

  <img src=".\assets\kernels.png" alt="Kernels" style="zoom:80%;" />

- The above values are written by human and it was done before 2012. 

- Now the network itself determines appropriate values.

- -1 suppresses the value

- 2 is used for amplification

- The values at the top(-1) will be negated

- The values in the middle will be amplified by double.

- The values in the bottom(-1) will also be negated

- The only thing that can be done is identification of middle row for a horizontal line.

  <img src=".\assets\vertical-edge-detector.png" alt="Vertical Edge Detector" style="zoom:80%;" />

- Values near to 1 are dark and near to 0 are white.

- We can see there is a vertical edge (0.9, 0.9, 0.8, 0.9, 0.9)

- MP reduces the dimension of an image

- With MP, we doesn't loose the features.

- MP does the job of filtering things.

- We never use 3x3 kernel in MP, we always use 2x2 kernel

- We use MP to reduce the size of channels 

- MP speeds up the training

- We don't need any parameters for calculating Global Average Pool

- After 7, we are going to add all channels

- For an image of 224x224, we should stop at 7

- For an image of 32x32, we should stop at 5

- For an image of 28x28, we should stop at 3

- Why do we stop at Receptive field of 11 and do MP?

  - Near 11, network starts to see edges and gradients

  - Pick any area in an image and we'll see size of 11 is most appropriate for extracting information.

    <img src=".\assets\receptive-field-11.png" alt="Receptive field of 11" style="zoom:80%;" />

- Padding of 1 retains the size of image. Size of input == size of output

- Image size we work are generally of 224.

- Companies also blindly uses 224.

- Most of the network start at 56

- Job of first part of network(HEAD) is reduce the size of image. Use MP to reduce the size to 56 first

- Face recognition happens at 48. 

- Why 11x11 or 9x9 is totally dependent on your dataset.

- Images in cifar dataset is very small (32x32), thats the actual size of image

- Time: 1:18:10 --- 7x7 or 5x5 is a good size to identify the objects and apply MP.

  - How can we apply MP on 7x7 or 5x5?

- By this time, network might have discovered edges and gradients and they are ready to form channel and if we add MP now, kernel is forced to give an amplitude or learn an amplitude in which they are able to pass that particular feature to the next available layer and next layer can come in and then they can start making those features.

- Generally we carry forward similar jumps. Right now we are doing 11x11 of 5 layer. That block is going to be repeated again and again.

- So whatever jump we had in receptive field is followed for next layer also.

- Right now what was told is a lie, but very close to truth.

- MP doubles the RF

- Blocks are a group of layers such as 1>3>5>7>9>11

- Block is going to take channel and perform 5 convolutions and give an output.

- If required we can add an option of padding = true. this is to ensure size of output == size of input

  <img src=".\assets\receptive-field-blocks.png" alt="Receptive Field Block" style="zoom:80%;" />

- Do not use MP in the last when network is going to predict

- Use MP as far as possible from last layer of prediction..

  <img src="D:\Praveen\SourceControl\github\praveenraghuvanshi1512\EVA4\Session-2\assets\pre-post-mp.png" alt="Pre and Post MP" style="zoom:80%;" />

- If we do MP one more time, it'll reduce the image much smaller.

- Further MP will make an 8 to smaller 8 and later to a circle which will lead to wrong prediction. It'll become a single blot.

- Do not use MP close to final output

- We are going to take an image, perform convolution and max pooling till the RF is equal to size of the features of edges and gradients we can find for which we need to look at the dataset, that happens to be 9 or 11. After that we repeat the convolution and MP in a block 

- We'll stop and doesn't perform MP after 4th block and just before last layers

- It is going to destroy the information as channel size is very small.

- For every layer in a model, we must write output and RF irrespective of the framework used(Tensorflow, keras, caffe etc)

- If image size is 4x4 and we perform convolution on it, god know what is happening there.

- On a 4x4, even after padding, we are going to get output of 4x4.

- We need to ensure last channels are not very small. They must be atleast 5x5 for CIFAR, 7x7 for any other dataset. Atleast give 49 pixels for the last layer to sit and look at.

- MP adds a bit of shift, rotational and scale invariance

  <img src=".\assets\MP-shift-invariant.png" alt="Max Pooling Shift Invariant" style="zoom:50%;" />

- Invariant means data has changed but it does not matter.

- For large invariance, we need to use kernels such as deformable kernels.

- Rotational Invariance

  <img src=".\assets\MP-rotational-invariant.png" alt="MP Rotational Invariant" style="zoom:50%;" />

  

- Scale Invariance

  <img src=".\assets\MP-scale-invariant.png" alt="MP Scale Invariance" style="zoom:33%;" />

- In above figure, we are looking at vertical edge which is convolved by a vertical kernel.

- On convolving a vertical edge with a vertical kernel, we expect a vertical edge output

- The kernel which extract a vertical line cannot extract a 45 degree line.

- 1 Kernel gives 1 Channel

- We have used 32 kernels

- Every kernel is different and creates its own channel.

- On using 32 kernels, we expect 32 channels to be generated.

  - 32 in middle slot is 32 channels

  400 x 400 | 3 x 3 x 32 | 398 x 398 x 32

  <img src=".\assets\Network-layers.png" alt="NN layers" style="zoom:33%;" />

- How many features we are seeing at this layer? 512

- How many channels are there at this layer ? 512

- 512 is a big no of channels

- 512 is not a large number feature

- 512 is a small number feature, not yet fit for medical data

- In medical data we would want to go to large number such as 1024 and 2048 etc.

- 512 is the number that determine

- 512 is the capacity of model

- 512 is very expensive

- Raspberry pi we may just need 24

- This no (512) depends on dataset

- This no(512) does not depend on images in a dataset.

- If we add million images of similar type, there won't be any change 

- Homogeneity of data should be understood

- If model is trained for cars and a new type of car is introduced by tata, we need not re-train and add kernel

- However, if image type is different such as model was trained for Tata cars and later Truck was introduced and added, we might have to re-train with new set of kernels which can detect big tyres, big windows etc. 

- Adding a new type/class will lead to re-training the model

- It depends on the way we are increasing the data.

- 512 is a safe no, it's the no where our network starts to learn something

- For raspberry pi we need to reduce this no to 24

  <img src=".\assets\network-comparison.png" alt="Network comparison" style="zoom:33%;" />

- 1 kernel == 1 Channel(always) 

- A horizontal edge detector kernel will always give a horizontal line

- Never divide channel into inputs

- No of output is 1.

- Each output may have multiple channels

- MP is allowing us to reduce the size in x and y

  <img src=".\assets\channel-explosion.png" alt="Channel explosion" style="zoom:67%;" />

- 3 x 3 x 512 x 512 =  2,359,296 (2.3 million) is an insane no of parameters. This much will be there in the RAM

- YOLO is around 16-17 million

- Resnet is around 21 million

- EfficientNet is around 7 million

- After each convolution we need an activation function

- Run the network for only 1 epoch for this assignment

- There is something buggy in the code, we need to fix it while attempting the quiz.

- Edges - sharp pixel changes
  Gradient - slow pixel changes
  Change = DELTA also called gradient. Don’t confuse this change/delta/gradient to above Gradient.




## Assignment

- [PyTorch NN tutorial](https://pytorch.org/tutorials/beginner/nn_tutorial.html)
- [Build PyTorch CNN - Object Oriented Neural Networks](https://www.youtube.com/watch?v=k4jY9L8H89U)
- https://www.aiworkbox.com/lessons/how-to-define-a-convolutional-layer-in-pytorch
