# EVA4 Assignment 1 - Praveen Raghuvanshi

### Team

- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)
- Abhinav Rana
- Chandan Kumar
- Prachi
- Harsha Vardhan

### Q & A

1. **What are Channels and Kernels (according to EVA)?**

   A channel comprise of a container of same or similar context. It represents a group of similar/same entities present in a image. Channels when combined produces an image. For e.g Red channel is formed by extracting all the red information from an image. An analogy is of sound recording where we have channel for each type of instrument such as drum, piano, voice etc. Things can be tweaked easily using different channels.

   A kernel is a matrix comprising of feature which when convolved over an image generates a feature map. It could be for detecting an eye, nose, etc in a face.

   The output of a kernel is a feature and feature map by retrieved collecting all such similar features from the image. One feature map forms a channel

2. **Why should we (nearly) always use 3x3 kernels?**

   Its a feature extractor and reduces dimensionality by 2. We can make kernel of any size using 3x3. It is symmetric compared to 2x2 which is not. Using 3x3 requires less parameters compared to kernels of other size such as 5x5 or 7x7. Using 3x3 twice is equivalent to single 5x5, however parameters in case of 3x3 is 18 and 25 in case of 5x5.

3. **How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)**

   199 x 199 > 197 x 197 | --> 1

   197 x 197 > 195 x 195 | --> 2

   195 x 195 > 193 x 193 |  --> 3

   193 x 193 > 191 x 191 |  --> 4

   191 x 191 > 189 x 189 | --> 5

   189 x 189 > 187 x 1897| --> 6

   187 x 187 > 185 x 185 | --> 7

   185 x 185 > 183 x 183| --> 8

   183 x 183 > 181 x 181 | --> 9

   181 x 181 > 179 x 179| --> 10

   179 x 179 > 177 x 177 | --> 11

   177 x 177 > 175 x 175 | --> 12

   175 x 175 > 173 x 173 | --> 13

   173 x 173 > 171x 171 | --> 14

   171 x 171 > 169 x 169 | --> 15

   169 x 169 > 167 x 167| --> 16

   167 x 167 > 165 x 165 | --> 17

   165 x 165 > 167 x 167 | --> 18

   163 x 163 > 161 x 161 | --> 19

   161 x 161 > 159 x 159| --> 20

   159 x 159 > 157 x 157 | --> 21

   157 x 157 > 155 x 155 | --> 22

   155 x 155 > 153 x 153 | --> 23

   153 x 153 > 155 x 155 | --> 24

   151 x 151 > 149 x 149 | --> 25

   149 x 149 > 147 x 147 | --> 26

   147 x 147 > 145 x 145 | --> 27

   145 x 145 > 143 x 143 | --> 28

   143 x 143 > 141 x 141 | --> 29

   141 x 141 > 139 x 139 | --> 30

   139 x 139 > 137 x 137 | --> 31

   137 x 137 > 135 x 135| --> 32

   135 x 135 > 133 x 133 | --> 33

   133 x 133 > 131 x 131 | --> 34

   131 x 131 > 129 x 129 | --> 35

   129 x 129 > 127 x127 | --> 36

   127 x 127 > 125 x 125 | --> 37

   125 x 125 > 123 x 123 | --> 38

   123 x 123 > 121 x 121 | --> 39

   121 x 121 > 119 x 119 | --> 40

   119 x 119  > 117 x 117 | --> 41

   117 x 117 > 115 x 115  | --> 42

   115 x 115  > 113 x 113 | --> 43

   113 x 113 > 111 x 111 | --> 44

   111 x 111  > 109 x 109 | --> 45

   109 x 109 > 107 x 107  | --> 46

   107 x 107 > 105 x 105 | --> 47

   105 x 105 > 103 x 103 | --> 48

   103 x 103 > 101 x 101 | --> 49

   101 x 101 > 99 x 99 | --> 50

   99 x 99 > 97 x 97 | --> 51

   97 x 97 > 95 x 95| --> 52

   95 x 95 > 93 x 93 | --> 53

   93 x 93 > 91 x 91 | --> 54

   91 x 91 > 89 x 89 | --> 55

   89 x 89 > 87 x 87 | --> 56

   87 x 87 > 85 x 85 | --> 57

   85 x 85 > 83 x 83 | --> 58

   83 x 83 > 81 x 81 | --> 59

   81 x 81 > 79 x 79 | --> 60

   79 x 79 > 77 x 77 | --> 61

   77 x 77 > 75 x 75 | --> 62

   75 x 75 > 73 x 73 | --> 63

   73 x 73 > 71 x 71 | --> 64

   71 x 71 > 69 x 69 | --> 65

   69 x 69 > 67 x 67 | --> 66

   67 x 67 > 65 x 65 | --> 67

   65 x 65 > 63 x 63 | --> 68

   63 x 63 > 61 x 61 | --> 69

   61 x 61 > 59 x 59 | --> 70

   59 x 59 > 57 x 57 | --> 71

   57 x 57 > 55 x 55 | --> 72

   55 x 55 > 53 x 53 | --> 73

   53 x 53 > 51 x 51 | --> 74

   51 x 51 > 49 x 49 | --> 75

   49 x 49 > 47 x 47 | --> 76

   47 x 47 > 45 x 45 | --> 77

   45 x 45 > 43 x 43 | --> 78

   43 x 43 > 41 x 41 | --> 79

   41 x 41 > 39 x 39 | --> 80

   39 x 39 > 37 x 37 | --> 81

   37 x 37 > 35 x 35 | --> 82

   35 x 35 > 33 x 33 | --> 83

   33 x 33 > 31 x 31 | --> 84

   31 x 31 > 29 x 29 | --> 85

   29 x 29 > 27 x 27 | --> 86

   27 x 27 > 25 x 25 | --> 87

   25 x 25 > 23 x 23 | --> 88

   23 x 23 > 21 x 21 | --> 89

   21 x 21 > 19 x 19 | --> 90

   19 x 19 > 17 x 17 | --> 91

   17 x 17 > 15 x 15 | --> 92

   15 x 15 > 13 x 13 | --> 93

   13 x 13 > 11 x 11 | --> 94

   11 x 11 > 9 x 9 | --> 95

    9 x 9 > 7 x 7 | --> 96

   7 x 7 > 5 x 5 | --> 97

   5 x 5 > 3 x 3| --> 98

   3 x 3 > 1 x 1 | --> 99

4. **How are kernels initialized?** 

   Kernels are initialized randomly. They are determined by network and can comprise of different kernels such as vertical and horizontal lines, light and gray textures etc. Before 2012 scientists used to do it manually and now its done automatically by the network. 

5. **What happens during the training of a DNN?**

   During the training, network tries to extract features by convolving different kernels over the image. These features are extracted in the order of edges and gradients, textures and patterns, parts of objects and objects finally. Network also adjusts weight during a process called backpropagation and its done by increasing/decreasing the gradients.

### References

- [**Introduction to Computer Vision with Deep Learning: Channels and feature maps**](https://mc.ai/3-introduction-to-computer-vision-with-deep-learning-channels-and-feature-maps/)
