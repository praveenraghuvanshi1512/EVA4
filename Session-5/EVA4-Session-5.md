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

- ### **[CODE3 - LIGHTER MODEL](https://colab.research.google.com/drive/1t0jdeu4Rg-GRPm2RNs7q1-MvA_3uCPyW)**

- ### **[ CODE4 - BATCHNORM ](https://colab.research.google.com/drive/12rQ81lvZSVuVJNLZPKEXcpEzpj1yG304)**

- ### **[ CODE5 - REGULARIZATION ](https://colab.research.google.com/drive/1Go7RjeKO_vfpwrL5iASjRqRckYdIarMu)**

- ### **[ CODE6 - GLOBAL AVERAGE POOLING ](https://colab.research.google.com/drive/1sdrerGJCxke700Rm8HsAn67Qno10sdQc)**

- ### **[ CODE7 - INCREASE CAPACITY ](https://colab.research.google.com/drive/1TYGkW7UI_yEiHnKM7EpqWOPreNlGzohA)**

- ### **[ CODE8 - VOILA! ](https://colab.research.google.com/drive/1UZgYzHP_nQfh5o6EUu6XLY0CE-UpjWZW)**

- ### **[ CODE9 - IMAGE AUGMENTATION ](https://colab.research.google.com/drive/1Pm5XDZ_lwfQUbV30UpacmOcj0_Xb___K)**

- ### **[ CODE10 - OVERKILL ](https://colab.research.google.com/drive/1s8m6WQbR88u9B9981e-iy4JlUCppG1mG)**

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

- 1.11



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