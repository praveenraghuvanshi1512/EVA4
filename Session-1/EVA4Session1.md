# EVA4 - Session 1 : 15-Jan-2019

#### Video  : https://youtu.be/aR3WboFNVj0

[![EVA Session 1](http://img.youtube.com/vi/aR3WboFNVj0/0.jpg)](https://youtu.be/aR3WboFNVj0)

### Links

- [Eye Painting time lapse](https://youtu.be/jC6qegT972c)
- [Playing with layers](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)
- [Assignment -  Python 101](https://colab.research.google.com/drive/1lnw5tyDde1ldBn5TUn6OhpNS3AY5D4fK)

#### Notes

- Brain is magical, it fills color for us. Illusion

  <img src=".\assets\illusion.png" alt="Illusion" style="zoom: 67%;" />

- Color is not going to be the main component for feature extraction in DNN

- DNN are not going to learn colors.

- For E.g, consider an apple or banana. What color they have? Red, green, etc... Color doesn't specify its an apple or banana.

- Humans use 3 channels(RGB), a newspaper use 4 channels(CYMK).

- We are not limited to only 3 channels(RGB), it's we decide on how many channels to be used. We called them as Red, Green and Blue

- Magazines are printed with 6 channels. A painter can paint with millions of colors(channels)

- We can take information and divide it into as much detail as possible

  <img src=".\assets\rgb-cmyk.png" alt="image-20200119000401779" style="zoom: 33%;" />

- We can't focus on color only such as Red in a image. Our focus will digress on other objects.

- DNN focus on features/objects and not on the color

- Songs recording comprise of multiple channels. There is a channel for voice, instrument(drum, guitar)

- We can increase/decrease the intensity of these channels.

- We must divide our problem into different channels so that we can have better feature extraction and tweak it well.

- Look at below image and focus on letter 'e' only.

  <img src=".\assets\feature-letter-e.png" alt="Feature Letter e" style="zoom:33%;" />

- Channels Clip: 39.30 - 45.45

- Alphabets : 26

- If we are looking at 'e', it means its a 'e' channel.

- If we are looking at 'a', it means its a 'a' channel.

- What is an 'e' channel? It's where all 'e's are visible. 

- 'e' itself is a feature

- In above image, 26 channels are visible.

- Imagine old projectors where transparent sheets were used.

- Suppose there is a stack of 26 sheets used to form an image

- Each sheet is a channel.

- The guy going to extract 'e' is called as a filter OR feature extractor OR a kernel(3x3)

- A kernel/filter/feature extractor is going to extract a feature for us

- If in a image, we have many 'e's, a kernel is going to extract all 'e's

- When a kernel/filter/feature extractor works on top of a image, its going to create a channel.

- To extract 26 channels here, we need 26 kernels

- Name one channels where all animals come together: Animal Planet

- Each channel is different such as Discovery, Animal planet, CNN, Fox News etc.

- We understand the context of a channel from some of its peculiar behavior.

- Some of the things maybe similar in most of the channels such as Ads, people moving here and there etc, still we are able to differentiate them

- A channel is a container of same or similar context.

- We are not able to call something a channel until and unless it contains things of similar context.

- Kernel could be for detecting a eye, nose, ear in a face.

- Once we extract these simple things such as an eye, nose, etc and combine to form a complex channel.

- The power of 26 alphabets(kernel) could allow us to understand lot of things such as history of world war, game of thrones etc. The possibilities with 26 alphabets are enormous.

- Consider an image being 32 alphabets(kernel), the features extracted will be very much.

- Alphabets are combined to form words -> sentences -> paragraphs -> stories -> books

- Same in image domain, we start with edges and gradients ->  combine them to form textures -> patterns -> part of objects -> objects -> scenes. These are the layers we need

- Our brain has 4 layers.

- Every single modern network has 4 blocks. Block 1, Block 2, Block 3 and Block 4.

- Purpose of 1st block : Extract edges and gradients

- Purpose of 2nd block : Extract texture and patterns 

- Purpose of 3rd block : Extract parts of objects

- Purpose of 4th block : Extract objects

- Cat Experiment

  <img src=".\assets\cat-experiment.jpg" alt="Cat Experiment" style="zoom: 67%;" />

- If a particular neuron fires for a particular edge, it never fires for others.

- For e.g if a neuron fires for a vertical edge or 35 degree edge, it never fires for it again.

- The image that is getting printed on the brain is getting printed on those neurons which are edge detectors

- This is the start of CNN or computer vision

- Laplacian, Sobel X and Sobel Y are Edge Extractors 

  <img src=".\assets\feature-extractors.jpg" alt="Edge Extractors" style="zoom:80%;" />

- In Sobel X, we are extracting edges on X-axis

- In Sobel Y, we are extracting edges on Y-axis

- We can take these small things and combine them to form complex images

- How many different kind of edges we need to form this digit?

  <img src=".\assets\digits.jpg" alt="Digits" style="zoom: 33%;" />

- We need only 2 of them (Vertical and horizontal)

- In order to reconstruct above image, we need more kernels such as vertical, horizontal, curve, etc.

- Interestingly, we only need 32 features/kernels to start with and 500 for every thing in this world.

  <img src=".\assets\stages-image-processing.png" alt="DNN stages" style="zoom:80%;" />

- Dog Nose Extractor

  <img src=".\assets\feature-nose-extractor.jpg" alt="Dog-Nose-Extractor" style="zoom:33%;" />

- If we want to extract a circle from an image

- We need to have a circle extractor(kernel)

- DNN is going to extract strokes which make sense

- We as humans will restrict on the no of strokes. for e.g limit to 500 strokes only.

- We need to tell our network that from 500 strokes you are only allowed to combine 500 strokes to form parts of objects. These are the no which we need to tell to our network. DNN takes this no and utilize to its capacity to use in the network and give back the image

- We can build very complex things by using small no of features/alphabets(26)

- In case of English, 26 letter can be used to form complex things. Similarly, in case of images, limited no of edges and gradients can be used to form complex images.

- We have 4 layers in the brain.

  <img src=".\assets\brain-4-layers.png" alt="Brain 4 Layers" style="zoom:80%;" />

- When we look at a Fish, we need to look as well as say its a Fish.

- Information is processed Outside In

  <img src=".\assets\dnn-layers.png" alt="DNN Layers" style="zoom:80%;" />

- Four layers: Edges and Gradiens -> Textures and Patterns -> Parts of objects -> Objects

- For initial sessions < 5, there is one restriction that is Size of image == Size of object

- Image of a dog(400x400) filled with dog only

- These are the kernels(96) of our CNN and its generated by DNN similar to 26 alphabets

  <img src=".\assets\cnn-kernels.jpg" alt="CNN kernels" style="zoom:80%;" />

- These are Features and NOT channels

- They are not kernels

- Someone has extracted this information. Whoever has extracted is a kernel

- We are just looking at a feature and can't make sense out of it

  <img src=".\assets\cnn-kernels-alphabets.jpg" alt="CNN Kernels Alphabets" style="zoom:80%;" />

- If we take vertical feature from above image and convolve over Crossword, we'll get bottom-left image

- If we combine all of them, we'll get the image back

- AS we move up, we need to increase the no of features.

- Going from 32 to 64 is not doubling, its a exponential growth.

- The combination increases exponentially

  <img src=".\assets\feature-engineering.JPG" alt="Feature engineering" style="zoom:67%;" />

- We are going to tell DNN that we are providing 32 features only and DNN is going to look at the dataset and try and get the best possible results.

- If we increase from 32 -> results will change.

- We are interested in combination of features and not only the features

- The interaction of us with computer science is Deep engineering. How many features our networks should have?

- The network is going to extract 32 best features from entire dataset

-  Where is the 3D version of link? 

- **Revisit Play link clip**

- CNN

- What is Receptive field? It should have seen the whole image

  <img src=".\assets\receptive-field.JPG" alt="Receptive Field" style="zoom:80%;" />

- Local receptive field == size of kernel (3x3)

- Global receptive field == 5x5 as it has seen whole image

- 3x3 convolved with 3x3 gives 1

- 5x5 convolved with 5x5 gives 1 = 25 parameters

- 5x5 convolved with 3x3 gives 3x3 --> Now 3x3 convolved with 3x3 gives 1

  - (3x3) + (3x3) = 18 parameters. less than 25
  - When we use 3x3 to get the same RF, parameters required is less
  - It reduces no of parameters required and computation
  - Its fast
  - Everyone is on 3x3
  - 3x3 has a symmetry and center line which 2x2 doesn't have
  - Every 3x3 is 2x2 also
  - No of parameter == kernel size(3x3) == 9
  - 3x3 might have a RF of 41

- Addition of a layer increases receptive field by 2

- Consider 5x5 is the output, then input must have been 7x7

- Similarly for 7x7 output, input must have been 9x9

- If we are convolution from 9 -> 7 -> 5 -> 3, then receptive field would be 9

- Image itself has a RF of 1. Each pixel knows about itself.

- Initial 4 sessions, stop at the size of image

- 200 layers is an insane no of layers, you need to be a billionaire to run such a large network

- our brain has 4 layers

- Modern networks have 50-100 layers

- Difference from left and right pixel is a gradient. Its a mathematical representation.

- The gradient which we are talking about is change in color. It is measurable.

  Zero gradient

  <img src=".\assets\gradient-zero.png" alt="Zero Gradient" style="zoom: 33%;" />

  <img src=".\assets\gradient.png" alt="Gradient" style="zoom:50%;" />

- FC layers destroys image information such as localization, contour, etc.

- 11x11 is golden at which we start to see something. 