# EVA4 - Session 5 : 12-Feb-2020

#### Video  : https://youtu.be/RQvubqNIqXw

[![EVA-4 Session 5](http://img.youtube.com/vi/RQvubqNIqXw/0.jpg)](https://youtu.be/RQvubqNIqXw)

### Links

- [Session-5 Pdf](S5.pdf)

- [Video](https://youtu.be/RQvubqNIqXw)

- [Assignment](https://colab.research.google.com/drive/1uJZvJdi5VprOQHROtJIHy0mnY2afjNlx)

- ### **[ CODE 1 - SETUP ](https://colab.research.google.com/drive/1aFgWmHNJoCyZ56zRvoE8xUdAe285aWmb)**

  - Converting to Tensor and Normalizing are the basic transforms we need to have.

- ### **[ CODE 2 - BASIC SKELETON ](https://colab.research.google.com/drive/1zx12oDfnadaVjEwQfUtAwfCQTSqZxRwj)**

  - Get the basic skeleton right

  - Target is NOT accuracy/parameters

  - We need to decide where to put Max Pooling(MP), 1x1, overall structure, Receptive Field(RF) is calculated at this point

  - No Fancy Stuff

  - Just one change

  - We are going to make only one change for which we are confident enough

  - There is only one change and that is in model

  - Refactored model into reusable blocks

  - Added nn.sequential for changing things

  - Created different Conv blocks

  - Sequential model with conv followed by ReLU

  - Keep bias outside

  - Need to write RF

  - Decided to do MP after 7x7

  - Transition block(Reducing channels using 1x1 from 128->32)

  - Following compression model

  - We are following image size

  - Once we reach image size of 7x7, we realize that we can't have MP anymore

  - We added big kernel of size 7x7

  - No relu at the last layer

  - We reduced no of channels to 10 as it the no of classes

  - This is the basic model setup and we'll twak this

  - Code is much easier to read now

  - Parameters is 1.94M

  - Look at logs

    - We should look at these no and do analysis
    - Is this model overfitting
    - Epoch 17: Test acc: 99.27 and Train Acc: 99.02
    - Calculate overfitting: (100 - Test Acc) + Train Acc > 99.4(SOTA)
    - (100 - 99.02) + 99.27 = 0.98 + 99.02 = 100 > 99.4 --> No Overfitting

    

- ### **[CODE3 - LIGHTER MODEL](https://colab.research.google.com/drive/1t0jdeu4Rg-GRPm2RNs7q1-MvA_3uCPyW)**

  - Target is to make the model light
  - Reduce no of parameters
  - We need to make the model as light as possible
  - Again only one change in the Model
  - Change in no of channels to reduce the parameter.
  - Reduce channels to as low as possible
  - Big Kernel(7x7) to convert channels to 1x1
  - No of parameters reduced drastically from 1.49M to 10,790
  - Train Acc : 99.0 and Test Acc: 98.98
  - (100 - 99.0) + 98.98 = 1 + 98.98 = 99.98 > 99.4 --> No overfitting
  - These are very good numbers as they are very close
  - When we see a model with such number we should be happy as no's are very good with slight difference between train dna test accuracy. 99 - 98.98 = 0.2
  -  Model doesn't have good capacity to push further
  - No overfittinf
  - The model can do great if trained for large no of epochs(30, 40, 50) and it may hit 99.4
  - This is unique scenario where we need not add Regularization, BN etc.
  - MNIST is a very easy dataset
  - Model is capable if pushed further
  - 

- ### **[ CODE4 - BATCHNORM ](https://colab.research.google.com/drive/12rQ81lvZSVuVJNLZPKEXcpEzpj1yG304)**

  - Capacity can be added in 2 ways
  - One way of adding capacity is by getting the data correct
  - To achieve this, we use a technique called as BN
  - After BN, we know all the kernels are going to get channels which are in specific range
  - If we use BN, all the no's are going to be kept between +3 and -3.
  - the only difference between code 3 and 4 is BN(Just one line difference)
  - We need to ensure that no of channels in BN is equal to the one in preceding layer
  - Should BN be done before/after ReLU? Doesn't matter and will be discussed in next session
  - Use BN for every single layer
  - With BN, model capacity increases and it trains faster because each kernel has to play in a small domain as compared to large predictions
  - Adding BN has increased parameters by 200
  - We also see trainable & non-trainable parameters
  - Results: Train - 99.9  and Test: 99.2
  - With BN, acc has increased
  - We were stuck at 99.0 earlier
  - We need to understand the effect of using something
  - We have hit 99.3 test accuracy
  - Train acc is 99.0, a diff of 0.3 which is quite less and good. No overfitting
  - We are just 0.1 away from target of 99.4
  - Overfitting 
  - 

- ### **[ CODE5 - REGULARIZATION ](https://colab.research.google.com/drive/1Go7RjeKO_vfpwrL5iASjRqRckYdIarMu)**

  - We are happy with model
  - But there is overfitting happening
  - Now we have to go for regularization
  - Adding dropout, image augmentation should be done only when we see this disease of overfitting
  - We have exhausted BN, parameters
  - Now, we should go with Regularization
  - Adding BN is also kind of Regularization as it is going to reduce the gap between test and train acc.
  - What is Regularization?
    - It is not increasing training Acc
    - It is not increasing test Acc
    - It is not making model more efficient
    - It is about Reducing the gap between test and train accuracy
    - The task of any regularization correctly is to reduce the gap between test and train acc
  - Dropout is one of the regularization technique
  - Regularization job is used to ensure not everyone is used in the network.
  - Regularization is like server backup
  - Task of regularization is to change the loss slightly
  - When we are stuck with acc, it goes to local minima/plateau
  - Using dropout of 0.5, 0.9 randomly is not going to help us
  - We need to get out of local minima
  - We don't use Regularize the techniques all at the same time, the model will superimpose and loss will be zig-zag, we don't do it
  - Choose medicines appropriately
  - Dropout is one such
  - Regularization will make the model efficient and not going to make it zero by making particular channel as zero. 
  - Pixel values will become zero
  - Later on we'll see L1 and L2 where weights become zero
  - We are not going to add dropout at a particular layer
  - Model capacity has reduced
  - Regularization doesn't necessarily reduce the gap, but it reduces the capacity of model.
  - There is bottleneck now with the introduction of Regularization at random layers.
  - Add dropout at every later except last -1 layer.
  - There is a mistake of using a big size kernel (7x7) in layer
  - We are not using GAP also.

- ### **[ CODE6 - GLOBAL AVERAGE POOLING ](https://colab.research.google.com/drive/1sdrerGJCxke700Rm8HsAn67Qno10sdQc)**

  - We are just adding GAP layer by replacing 7x7 conv and reducing no of parameters
  - GAP, we used kernel of 7.
  - No of params has reduced from 10k to 6k
  - GAP doesn't take parameters
  - The logs are disappointing now. Stuck at 98%, not hitting 99%
  - Is GAP a good thing or bad
    - Wrong question. Can't compare with a model after removing 4000 parameters and saying why acc is reducing?

- ### **[ CODE7 - INCREASE CAPACITY ](https://colab.research.google.com/drive/1TYGkW7UI_yEiHnKM7EpqWOPreNlGzohA)**

  - Increased capacity by increasing channels at some layer (10-32)
  - Increased param to 11k
  - Train - 99.28 and Test - 98.96
  - Location of MP is not correct
  - At RF 5x5, we start to see patterns forming.
  - Image analysis was wrong
  - How many pixels should we traverse to start seeing patterns
  - First MP is the most important MP layer
  - Adding capacity not weights
  - Closer analysis, we can move MP one layer up. Currently, we are doing at 7x7, we can do it at 5x5
  - IMP: Adding a capacity by adding a layer after GAP

- ### **[ CODE8 - VOILA! ](https://colab.research.google.com/drive/1UZgYzHP_nQfh5o6EUu6XLY0CE-UpjWZW)**

  - Increase model capacity at the end(add later after GAP)

  - Perform MP at RF of 5

  - Fix dropout, add it to each layer

  - Using small value of dropout of 0.1

  - 1x1 is FC if input pattern is 1-D

  - We have added capacity after the GAP

  - We are using 16 cues and converting it into 16 points, we are getting more cues

  - FC layers won't die, but they would be used after GAP

  - Model params increased to 13k

  - Being hard on data, acc started with 88.92/98.20

  - Its because of adding dropout at every layer

  - At 8th epoch reached 99.4%

  - Train Acc has not gone till 99.4

  - Very good model

  - No overfitting

  - Let's add mode capacity to make it more predictable/confident

    

- ### **[ CODE9 - IMAGE AUGMENTATION ](https://colab.research.google.com/drive/1Pm5XDZ_lwfQUbV30UpacmOcj0_Xb___K)**

  - Random Rotation(-7.0, 7.0)

  - Its added before converting to Tensor

  - We can't add after Normalization because we have no clue one what color would have been added

  - Only one change

  - No Change in parameters

    - Till 11 or 12th epoch, it has not done well as data is harder

  - Very good model

  - Steps for solving a CV problem

    - Set the base right

    - Set the model right

    - Look at the disease

    - If disease is overfitting 

      - Add something
      - BN to increase capacity
      - Then add regularization

    - Finally add image augmentation to take it closer

      

- ### **[ CODE10 - OVERKILL ](https://colab.research.google.com/drive/1s8m6WQbR88u9B9981e-iy4JlUCppG1mG)**

  - Add rotation, our guess is that 5-7 degrees should be sufficient

  - Add LR scheduler

  - MNIST is too smell for LRScheduler

  - Only one change

  - LRScheduler usage(step LR)

  - Step-size : 6

  - Eighth epoch : 99.45%

  - No improvement further

  - Designing models require discipline

    

#### Assignment

- 

#### Notes

- The denser the model, difficult it will be train the model

- Sparse networks are easier to train compared to dense.

  <img src=".\assets\dense-vs-sparse.jpeg" alt="Dense Vs Sparse" style="zoom:67%;" />

- Given a data in 1D, which is preferred Conv1D ot FC?

  - Drawback with FC is no of channels. They have only 1 channel. Its difficult to encode full information from channel 1. 
  - If we need segregation of information, we use Convolution as it can have multiple channel.
  - In case we are interested in one information, we can go with FC layer.

- Information can be inferred in two ways : 

  - Spatial: Distance
  - Temporal: time

- Overfitting

  - When the gap between Train and Test accuracy is large

- Play with Datatransformations

- Look at the data first and then apply transformations

- Remove ToTensor and Normalization from transformers and use it.

- Standardizing is converting to ToTensor().

- 



#### Questions

- Where is the channel information after reading an image and assigning it to a tensor?
- 
- 
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