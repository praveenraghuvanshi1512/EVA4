# EVA4 - Session 8 : 04-March-2020

#### Video  : https://youtu.be/LhOnn0wkH-E

[![EVA-4 Session 8](http://img.youtube.com/vi/LhOnn0wkH-E/0.jpg)](https://youtu.be/LhOnn0wkH-E)

### Links

- [Session-8 Pdf](S8.pdf)

- [Video](https://youtu.be/LhOnn0wkH-E)

  



#### Assignment

- 

#### Notes

- This session is about Receptive field
- Imagenet is dead now
- BN was not invented till 2014, it came in 2015. ResNet also came in 2015
- 2016 came the Ensemble which is a mixture of multiple networks
- 2017 SeNets came which is a slight variation of ResNet
- Today we have ResNet which can go to 1000 layers
- Whatever be the number of layers, we have 4 blocks and 3 MP 
- It means our receptive fields are sky-rocketing
- Imagine RF of networks with 1000 layers, its going to be huge
- All modern networks use Padding,
- MP is only area where size of image/channel changes
- Receptive field is linked to size of object
- If we have a RF of 1000 and wanted to see a small dog in it, it's of no use
- If we look at a RF of same size, that means we are looking with same eye
- It's not the best idea to look at things with same RF
- Till now we have written a network with single RF
- Now, we have network with multiple RF in the last layer
- Object in an image may not be centered.
- Size of object is not fixed and varies, means RF also varies
- There is a problem in Yolo where we have a skip connection linking 13th layer to the last layer.
- 13th layer might be able to analyze small objects which gets connected to last layer for prediction.
- If we take an example of Cancer cells which have same size, we just need single receptive field
- In case of camera capturing images of person, we need to have different RF.
- Multiple RFs allows identifying sharp features
- We need multiple RF whenever a precision is required.
- MP is taking an image and literally passing it on
- There is not ideal RF as we have multiple RFs
- In Modern architecture such as ResNet, we carry all the RFs to the last and make better predictions
- When we go beyond RF, we are making object templates
- If we go deep and without image augmentation, then we are doomed
- Most of the time we train a network to detect something
- Convolution happens outside the main layer
- Convolution is the residue in ResNet
- We don't do anything on main line
- Anything modification such as dropout will lead to lower accuracy
- Perform all modifications on the convolution path.
- The difference between ResNet V1 and V2 is in V2, they took Relu inside 
- In Resnet architecture there is only one MP.
- Resnet is designed to be fast
- Stride of 2 adds checkerboard issue, reduces memory
- Dotted line is a 1x1 with a stride of 2.



#### Questions

- Concatenation Vs Addition 
- Skip connection
- Residual module
- object templates
- vanishing and exploding gradients
- Identity layers
- Regular Vs Bottleneck ResNet

#### Further Study

- 

### References

- 