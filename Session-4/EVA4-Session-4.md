# EVA4 - Session 4 : 05-Feb-2020

#### Video  : https://youtu.be/OW30SxJAoAw

[![EVA-4 Session 4](http://img.youtube.com/vi/OW30SxJAoAw/0.jpg)](https://youtu.be/OW30SxJAoAw)

### Links

- [Session-4 Pdf](S4.pdf)
- [Video](https://youtu.be/OW30SxJAoAw)
- [Assignment](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx)

#### Assignment

- *lines are what all matter*
  - Lines are synapses which connects neurons together through a dendrite
  - Lines contains the weights/strength 
- *Exactly, that's the point*
  - A flattened array doesn't signify anything
  - It's generated in a fully connected layer
- *This is how weights and multiplications work in case of FC layers:*
  - Each and every input is connected to a output
- *Because we are summing the loss function to all the correct classes*
- *Fantastic Beasts and Where to Use them!*
- *indirectly you have sort of already used it!*
- 

#### Notes

- **Cutout** is the best augmentation strategy

- **GradCam** is used to identify where networking is looking within a image

- Bottleneck

  - Consider a layer with 3x3x256x512
  - We are not tagging anywhere in 256 channels of 3x3 that its a tiger.

- We loose spatial information when we use fully connected layer

- It's used to convert a 2D data to 1D. Never use it for this conversion. 

- FC layers can be used in 1D 

- FC is translation, rotational variant

- FC is full of lines/connections whereas convolution has few lines.

- In FC, there is no Local RF, there is always Global RF which is equal to size of input

- Even though convolution is sparse, the no of weights used is pretty low.

  <img src=".\assets\FC-conv.jpeg" alt="Fully connected vs Convolution" style="zoom:50%;" />

- 81 = 9x9, 225 = (5x5)x(3x3)

- ResNet is a good architecture and we are not using FC in it

- Performing GAP on 512, we get 512 numbers

- GAP converts channels into numbers

- GAP is not destroying the information, it is just amplifying the values

- Last layer has the object in it.

- After GAP, we can use FC in order to convert from 512 to 1000(no of classes)

- We haven't destroyed the information by using reshape such as 2D to 1D in FC layer

  <img src=".\assets\gap.jpeg" alt="GAP" style="zoom:50%;" />

- There is no way to change from 512 to 1000 classes, GAP helps achieve this.

-  1x1 is FC layer used in 1D

  <img src=".\assets\gap-fc.jpeg" alt="GAP vs FC" style="zoom:50%;" />

- FC counts to around 90% parameters in VGG16

- Do not use view to convert 2D to 1D. Use GAP

- Softmax is used to create gap between numbers

- Softmax helps make decisions

  <img src=".\assets\softmax.jpeg" alt="Softmax" style="zoom:50%;" />

- Confidence of each class needs to be high apart from accuracy.

- Softmax may hide that information/confidence

- log of softmax is preferred

- Softmax is likelihood and not the probability

- Probability is something when a probabilistic distribution is done, which is not in this case.

- For Cat and Dog, its not the probability that there is a probability of being a dog is 45%. We say, it's the features present in a dog that is making it 45%.

- The higher the loss, the unhappy is the network and vice versa

- We want value given by softmax to be large

- Log calculation is done using natural scale of exponential. Do not use scale of 10

  <img src=".\assets\negative-log-loss.jpeg" alt="Negative Likelihood Log Loss" style="zoom:50%;" />

- We need to calculate log of softmax of correctly predicted class.

- Batch Normalization helps amplify the features.

- We normalize values between 0 and 1.

- We are going to take a specific channel with a specific kernel for all the images in the batch and then normalize those channels.

- In dropout we disable some of the neurons randomly.

- Dropout is used for regularization

- We don't know which layer is overlearning.

- Add dropout to all layers with small amount

- Cutout is preferred over dropout

- During training only, we drop the neurons and preserve them during testing

- Dropout is preventing overfitting by removing some of the pixels which is unable to learn.

- Learning rate: Stick to SGD

- MNIST is different from others. Others have objects, it doesn't have objects. It have only digits.

- 4 blocks are not required for MNIST, only 2 blocks are enough.

- There is no textures, parts of objects and objects in MNIST.

- MNIST is very different and simple dataset.

- There is no point in going beyond receptive field

- Do not use anything before last layer

- 



#### Questions

- What is GAP? 
- How is it different from normal Pooling operation? 
- Does GAP also reduce no of parameters ?
- Does it also reduce image size by half?
- Does it double RF?
- What is 1x1 Convolution?
- Why is the significance of dropping from 512 -> 32 in next block? 
- What is log of softmax?
- Concept of Transition Layers,
- Position of Transition Layer,
- When do we introduce DropOut, or when do we know we have some overfitting
- The distance of MaxPooling from Prediction,
- The distance of Batch Normalization from Prediction

#### Further Study

- 1x1 convolution
- GAP
- Bottleneck
- Reverse Bottleneck
- Depthwise separable convolution
- 

### References

- 