# EVA4 - Session 11 : 05-April-2020

#### Video  : https://youtu.be/YE3BWPGPdn8

[![EVA-4 Session 11](http://img.youtube.com/vi/YE3BWPGPdn8/0.jpg)](https://youtu.be/YE3BWPGPdn8)

### Links

- [Session-11 Pdf](S11.pdf)

- [Video](https://youtu.be/YE3BWPGPdn8)

#### Assignment

- Topics
- Division Of work
  - Cyclic graph
  - Resnet architecture
  - One Cycle Policy
- Next meeting: Wednesday/Thursday 8 AM
  - Review
  - Challenges, communicate immediately
- Colab Link: https://colab.research.google.com/drive/1nGmlScyFQJ6kUGd9ER0WyfhWcEwqsiUN

#### Notes 

##### Video

-  We use regularization to not stuck in minima
-  If using one cycle policy, we may not require other regularization techniques such as Image augmentation, dropout L1/L2 and weight decay etc.
-  Absolute Vs Relative threshold
-  SGD looks at the depth and not the direction
-  It looks at the steepest minima
-  Best LR is for the weights at that point of time
-  LR will vary based on weights
-  One Cycle Policy(OCP) is designed to reach higher accuracy in less no of epochs.
-  Momentum is pushing in the direction where training is moving.
-  We are moving away from GAP and using MP equal to size of image of 4.
-  

##### Others

- 


#### Questions

- Why MP of 4 in last layer of assignment?
- 

#### Further Study

- Resnet Block
- Skip Connections
- Identity shortcut connections

### References

- [Training and investigating Residual Nets](http://torch.ch/blog/2016/02/04/resnets.html)
- [Deep learning residual networks](http://datahacker.rs/deep-learning-residual-networks/)
- https://medium.com/deepreview/review-of-deep-residual-learning-for-image-recognition-a92955acf3aa

#### Sample

- [https://github.com/mshilpaa/EVA4/blob/master/Session%2011/extras/s11.ipynb](https://github.com/mshilpaa/EVA4/blob/master/Session 11/extras/s11.ipynb)
- 



