# EVA4 - Session 10 : 18-March-2020

#### Video  : https://youtu.be/MQiM-tF0now

[![EVA-4 Session 10](http://img.youtube.com/vi/MQiM-tF0now/0.jpg)](https://youtu.be/MQiM-tF0now)

### Links

- [Session-10 Pdf](S10.pdf)

- [Video](https://youtu.be/MQiM-tF0now)

#### Assignment

- Topics
  - LR
  - Plateau
  - ReduceLR On Plateau
  - LR Finder
  - Momentum
  - Nesterov Momentum
  - Optimizers
    - SGD
    - Adam
    - RMSProp
    - Adagrad
  - Cyclic LR
- Division Of work
  - LR Finder
  - Plot Loss Curve
  - Finding LR
  - ReduceLROnPlateau
  - Plot Train and Test Acc
  - 
  - Share good references
  - Plot Acc on same graph
  - Finding misclassified images
- Next meeting: Sunday 8 AM
  - Review
  - Challenges, communicate immediately

#### Notes

-  Learning rate is about adjusting the weights of our network w.r.t loss of gradients.

- lower LR will make network slow

- For calculating the loss, our inputs(x1,x2,x3...) may be fixed, but weights(w1,w2,w3...) might change

- Loss is dependent on weights. L --> W

- Weight might not be moving in right direction

- The loss curve is not constant

  <img src=".\assets\loss_curve.png" alt="Loss Curve" style="zoom:50%;" />

- Gradient descent is we are looking for negative correlation

- Gradient ascent is we are looking for positive correlation

- Loss and weight are negatively correlated. 

- An increase in weight should reduce loss

- A decrease in weight should increase loss

- Weight has no control over the input. Input is constant

- Calculating loss is a very compute intensive operation

- Just for calculating a derivative of W1, we need to calculate derivative of W2, W3,W4...etc. Which is compute intensive considering we might have 1000's of weights.

- Derivative of W1 is 1.

- To overcome this , we use partial derivatives which has less computations.

- In partial derivates, we consider partial derivates of W2, W3, W4...etc to be very small, something near to zero.

  <img src=".\assets\derivative_partial.png" alt="Derivative vs Partial Derivative" style="zoom:50%;" />

- Alpha in above equation is used for balancing

- Finding alpha is very very difficult

- We reduce the learning rate by 10 every time in order to fix decimal places

- In Gradient descent, we scan through all items/images in a dataset and change the learning rate which can be compute intensive as in a dataset of face recognition where we can have millions of images

- To fix above issue, Stochastic Gradient descent came, which allows to take one image at a time and change learning rate. This leads to faster execution. The problem is changing LR at every image.

- To overcome SGD, mini-batch Gradient Descent is introduced which takes batch of images and updates the LR. Here we have a balance of LR update and Speed.

- Batch size selection has to be done carefully 

- For imagenet with 10000 classes, choosing a batch size of 10 is a very very bad idea.

- Gradient Perturbation is adding a small noise in order to move out of local minima.

- Test acc is going to be always more in case of SGD compared to Adam.

- SGD is not used when data is less

- SGD is not used in transfer learning

- SGD is not used in Reinforcement learning

- SGD is only used in object detection.



#### Questions

- How do you do X1 + X2 + X3 in S9 quiz?
- Why do we reduce LR by 10? 
- Why adding a comma in transformation is a good practice?

#### Further Study

- Partial Derivatives
- Stochastic Gradient Descent(SGD)
- Sparse Data
- Saddle Point
- Plateau
- Nesterov Momentum and plain momentum

### References

- 