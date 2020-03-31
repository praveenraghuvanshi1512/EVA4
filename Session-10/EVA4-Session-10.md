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

##### Video

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

##### Others

- **Back Propogation and Optimisation Function:** Error J(w) is a function of internal parameters of model i.e weights and bias. For accurate predictions, one needs to minimize the calculated error. In a neural network, this is done using back propagation. The current error is typically propagated backwards to a previous layer, where it is used to modify the weights and bias in such a way that the error is minimized. The weights are modified using a function called Optimization Function.

- **Adaptive Learning Algorithms:**

  The challenge of using gradient descent is that their hyper parameters have to be defined in advance and they depend heavily on the type of model and problem. Another problem is that the same learning rate is applied to all parameter updates. If we have sparse data, we may want to update the parameters in different extent instead.

  Adaptive gradient descent algorithms such as Adagrad, Adadelta, RMSprop, Adam, provide an alternative to classical SGD. They have per-paramter learning rate methods, which provide heuristic approach without requiring expensive work in tuning hyperparameters for the learning rate schedule manually.

- **Stochastic Gradient Descent** refers to an algorithm which operates on a batch size equal to 1

- **Mini-batch Gradient Descent** is adopted when the batch size is greater than 1.

- **Gradient Perturbation** : A very simple approach to the problem of plateaus is adding a small noisy term (Gaussian noise) to the gradient

- **Momentum and Nesterov Momentum**

  - A more robust solution is provided by introducing an [exponentially weighted moving average](https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average) for the gradients. The idea is very intuitive: instead of considering only the current gradient, we can *attach* part of its history to the correction factor, so to avoid an abrupt change when the surface becomes flat.
  - A slightly different variation is provided by the **Nesterov Momentum**. The difference with the base algorithm is that we first apply the correction with the current factor v(t) to determine the gradient and then compute v(t+1) and correct the parameters:

- **RMSProp** This algorithm, proposed by G. Hinton, is based on the idea to adapt the correction factor for each parameter, so to increase the effect on slowly-changing parameters and reduce it when their change magnitude is very large. This approach can dramatically improve the performance of a deep network, but it’s a little bit more expensive than Momentum because we need to compute a *speed* term for each parameter:

- **Adam** is an adaptive algorithm that could be considered as an extension of **RMSProp**. Instead of considering the only exponentially weighted moving average of the gradient square, it computes also the same value for the gradient itself:

- **Adagrad** This is another adaptive algorithm based on the idea to consider the historical sum of the gradient square and set the correction factor of a parameter so to scale its value with the reciprocal of the squared historical sum. The concept is not very different from RMSProp and Adam, but, in this case, we don’t use an exponentially weighted moving average, but the whole history. 

- **AdaDelta** is algorithm proposed by M. D. Zeiler to solve the problem of AdaGrad. The idea is to consider a limited window instead of accumulating for the whole history. In particular, this result is achieved using an exponentially weighted moving average (like for RMSProp)

- **Conclusion** 

  - Stochastic Gradient Descent is intrinsically a powerful method, however, in non-convex scenarios, its performances can be degraded. We have explored different algorithms (most of them are currently the first choice in deep learning tasks), showing their strengths and weaknesses. Now the question is: which is the best? The answer, unfortunately, doesn’t exist. All of them can perform well in some contexts and bad in others. In general, all adaptive methods tend to show similar behaviors, but every problem is a separate universe and the only silver bullet we have is trial and error. I hope the exploration has been clear and any constructive comment or question is welcome!

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
- More
  - Resnet
  - Concatenation and Addition
  - Skip Connections
  - 

### References

- [Loss Functions and Optimization Algorithms. Demystified](https://medium.com/data-science-group-iitr/loss-functions-and-optimization-algorithms-demystified-bb92daff331c)
- [A Brief (and Comprehensive) Guide to Stochastic Gradient Descent Algorithms](https://www.bonaccorso.eu/2017/10/03/a-brief-and-comprehensive-guide-to-stochastic-gradient-descent-algorithms/)
- [SGD vs Adam?](https://shaoanlu.wordpress.com/2017/05/29/sgd-all-which-one-is-the-best-optimizer-dogs-vs-cats-toy-experiment/)
- [An overview of gradient descent optimization algorithms](https://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose)
- [Week5: CIFAR-10 + Data Augmentation](https://wp.nyu.edu/shanghai-ima-documentation/uncategorized/yl4121/week5-cifar-10-data-augmentation/)
- [Data Augmentation](https://zhuanlan.zhihu.com/p/41679153)
- 



