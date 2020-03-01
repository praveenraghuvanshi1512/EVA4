# EVA4 - Session 7 : 26-Feb-2020

#### Video  : https://youtu.be/apr8pcKJY5k

[![EVA-4 Session 7](http://img.youtube.com/vi/apr8pcKJY5k/0.jpg)](https://youtu.be/apr8pcKJY5k)

### Links

- [Session-7 Pdf](S7.pdf)

- [Video](https://youtu.be/apr8pcKJY5k)

  



#### Assignment

- 

#### Notes

- Picking a proper value for L1 and L2 is an art

- There is a process to solve it and its called as Grid Search

- Concatenation Vs Addition. Which one to choose?

  ​	<img src=".\assets\add_vs_concat.jpeg" alt="Addition Vs Concatenation" style="zoom:33%;" />

- The answer is, we really don't know.

- Concatenation is preferred, however Resnet uses addition and it has beaten all the accuracies

- We are going with Resnet V1 and V2 followed by ResNext.

- We are going to cover SK and SGNet which is an addition on top of Resnet

- New paper came out called as EfficientNet which has an accuracy of 80% on imagenet

- In this course main focus will be on ResNet and not on DenseNet or Inception etc.

- Local receptive field is equal to the size of image

- If we are going to change the size of image with stride, no change on immediate layer, however its effect will be seen in consecutive layers

- Pointwise convolution is a fancy name for 1x1 which is used to reduce the no of channels

  ​	<img src=".\assets\pointwise_convlolution_1x1.jpeg" alt="Pointwise or 1 x 1 convolution" style="zoom:50%;" />

- We have learned that 1 x 1 should be used to reduce the no of channels. 

- It should not be used for increasing the no of channels if we have a 3 x 3 as a backup.

- It must be linked to 3 x 3 somehow.

- We cannot have two consecutive 1 x 1

- 1x1 used for increasing no of channels is done a lot 

- In object detection we should answer 'Where' and 'What'

- Dialated convolution or Atrous convolution

- Input image size must be same as output size

- It means we can't use MP, can't reduce the channels

- As channel size is going to be big, we can't use 3x3 to achieve this

- To fix this we use encoder-decoder architecture

- Latent vector, embeddings

- Image segmentation, pose estimation, 3-D convergence are the application of encoded-decoded

- Monocular depth estimation

- Colorization problems 

- Grayscale to Colored images

- Classifying each pixel

- Only job of dialated convolution is to predict what 3x3 is looking at, where the components are?

- In pytorch just specify dialation = 2

- Dialated convolution doesn't give sharp information.

- DC is used in deeper layers

- WaveNet uses concept of dialation

- Deconvolution OR Fractionally strided OR Transpose convolution

- What should be done to get 7x7 on convolving a 3x3 kernel over a 5x5. 

- The easiest way is padding but its inefficient

- Better approach is dialated convolution

- Instead of dialating the kernel, which increases receptive field, we can dialate our image.

- DC leads to checkerboard issue.

- In Checkerboard issue, some pixels emit out more information compared to others

- Un MP

- Depthwise separable convolution(DSC), a revolutionary convolution

- 8 times less parameters

- Ultra fast. used in Mobile architecture etc.

  ​	<img src=".\assets\depthwise-separable-convolution.jpeg" alt="Depthwise Separable Convolution" style="zoom:50%;" />

- Grouped convolution is an advanced convolution

- All networks after VGG uses grouped convolution

- GC is a fancy way of saying we have parallel convolutions and merging them later

- 1:56

#### Questions

- What is the difference between addition and concatenation?

#### Further Study

- What is Internal Covariate Shift?
- Pointwise convolution  or 1x1
- Checkerboard issue
- Pixel Shuffle
- Grouped convolution
- Latent vector, embeddings
- Encoding-Decoding architecture
- Monocular depth estimation

### References
