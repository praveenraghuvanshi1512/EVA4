# EVA4 Assignment 13 - Praveen Raghuvanshi

**Team Members**

- Tusharkant Biswal (Tusharkanta_biswal@stragure.com) 
- V N G Suman Kanukollu (sumankanukollu@gmail.com)
- Harsha Vardhan (harshavardhan.ma@gmail.com)
- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)

[Github Directory - Assignment -15-A](https://github.com/praveenraghuvanshi1512/EVA4/tree/master/Session-14/Assignment-15-A)

### Problem

<img src=".\assets\15-A-problem.jpg" alt="15-A-problem" style="zoom:67%;" />

 You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images. 

In total you MUST have:

1. 400k fg_bg images
2. 400k depth images
3. 400k mask images
4. generated from:
   1. 100 backgrounds
   2. 100 foregrounds, plus their flips
   3. 20 random placement on each background.
5. Now add a readme file on GitHub for Project 15A:
   1. Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. 
   2. Add your dataset statistics:
      1. Kinds of images (fg, bg, fg_bg, masks, depth)
      2. Total images of each kind
      3. The total size of the dataset
      4. Mean/STD values for your fg_bg, masks and depth images
   3. Show your dataset the way I have shown above in this readme
   4. Explain how you created your dataset
      1. how were fg created with transparency
      2. how were masks created for fgs
      3. how did you overlay the fg over bg and created 20 variants
      4. how did you create your depth images? 
6. Add the notebook file to your repo, one which you used to create this dataset
7. Add the notebook file to your repo, one which you used to calculate statistics for this dataset

 

Things to remember while creating this dataset:

1. stick to square images to make your life easy. 
2. We would use these images in a network which would take an fg_bg image AND bg image, and predict your MASK and Depth image. So the input to the network is, say, 224x224xM and 224x224xN, and the output is 224x224xO and 224x224xP. 
3. pick the resolution of your choice between 150 and 250 for ALL the images 

15A is a group assignment.

Questions asked in 15A:

1. Share the link to the readme file for your Assignment 15A. Read the assignment again to make sure you do not miss any part which you need to explain. -2500
2. Share the link to your notebook which you used to create this dataset. We are expecting to see how you manipulated images, overlay code, depth predictions. -250
3. Surprise question. -Surprise Marks. 

#### Solution

- This assignment is done as a GROUP and has same submission by all team members. 

- Team members are mentioned at the top

- Assignment has been executed on a Colab GPU machine.

- Answers to questions asked as part of assignment

- Now add a readme file on GitHub for Project 15A:

  1. Create this dataset and share a link to GDrive (publicly available to anyone) in this readme file. 

     - Suman - https://drive.google.com/drive/u/1/folders/1KY4jI1KesmIQzuwYpTfNX_d3epGMOXFB

     - Praveen - https://drive.google.com/drive/folders/1-2SntbxewUuyhll20FOz2egPX4dJ5y_Z?usp=sharing

       

  2. Add your dataset statistics:

     1. Kinds of images (fg, bg, fg_bg, masks, depth)
        - Folder names
          - Foreground -
          - Background - 
          - Foreground Flip - 
          - Foreground_background
     2. Total images of each kind
        - 
     3. The total size of the dataset
        - 
     4. Mean/STD values for your fg_bg, masks and depth images
        - 

  3. Show your dataset the way I have shown above in this readme

     - IMAGE

  4. Explain how you created your dataset

     1. how were fg created with transparency
        - 
     2. how were masks created for fgs
        - 
     3. how did you overlay the fg over bg and created 20 variants
        - 
     4. how did you create your depth images?
        - 

  

- **Submission**

  1. Share the link to the readme file for your Assignment 15A. Read the assignment again to make sure you do not miss any part which you need to explain. -2500
     - 

  2. Share the link to your notebook which you used to create this dataset. We are expecting to see how you manipulated images, overlay code, depth predictions. -250
     - 

  3. Surprise question. -Surprise Marks. 
     - Hopefully Rohan will bless us :-)