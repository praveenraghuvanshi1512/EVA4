# EVA4 Assignment 15-A - Praveen Raghuvanshi

**Team Members**

- Tusharkant Biswal (Tusharkanta_biswal@stragure.com) 
- V N G Suman Kanukollu (sumankanukollu@gmail.com)
- Praveen Raghuvanshi (praveenraghuvanshi@gmail.com)

[Github Directory - Assignment -15-A](https://github.com/praveenraghuvanshi1512/EVA4/tree/master/Session-15/Assignment-15/Assignment-15-A)

### Problem

<img src=".\assets\15-A-problem.png" alt="15-A-problem" style="zoom:67%;" />

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

     - Suman    - https://drive.google.com/drive/u/1/folders/1ZXdupMsfZLxgQ2dyToZ3rp6ukjNTMjY7
     - Praveen  - https://drive.google.com/drive/folders/1-2SntbxewUuyhll20FOz2egPX4dJ5y_Z?usp=sharing
     - Tushar   - https://drive.google.com/drive/folders/10t7wzl83M_g-CzKWoAjoVOLm6d9GIspL

       

  2. Add your dataset statistics:

     1. Kinds of images (fg, bg, fg_bg, masks, depth)
        
          
        
          | S.No | Type                                               | Count | Size      | Remarks                                                      |
          | ---- | -------------------------------------------------- | ----- | --------- | ------------------------------------------------------------ |
          | 1    | Background (BG) - JPG                              | 100   | 224 x 224 | - Library(50) + Classroom(50)                                |
          | 2    | Foreground (FG) - JPG                              | 100   | 96 x 96   | Dog breed (Chihuahua, golden_retriever, German_shepherd, Eskimo_dog) |
          | 3    | Mask - FG                                          | 100   | 96 x 96   |                                                              |
          | 4    | FG on BG + FG Flip on BG                           | 20000 | 224 x 224 |                                                              |
          | 5    | Random placement 1-FG on 1-BG at 20 locations      | 20    | 224 x 224 |                                                              |
          | 6    | Random placement 1-FG-Flip on 1-BG at 20 locations | 20    | 224 x 224 |                                                              |
        | 7    | BG-FG Images                                       | 400K  | 224 x 224 | Includes FG and FG-Flip                                      |
        | 8    | BG-FG Mask Images                                  | 400K  | 224 x 224 | Includes FG and FG-Flip                                      |
        | 9    | Dense Depth images for all BG-GF and BG-FG Flip    | 400K  | 224 x 224 |                                                              |
        | 10   | Total Images                                       | 1200K |           | 400K + 400K + 400K                                           |

     

     - Image Naming structure :
       
         | S.No | Format                 | Remarks                                                      |
         | ---- | ---------------------- | ------------------------------------------------------------ |
         | 1    | (x)_bg_(y)_fg          | on BG Image (x) FG Image (y) is overlaid                     |
         | 2    | (x)_bg_(y)_fg_mask     | on BG Image (x) FG Image (y) is overlaid with corresponding mask |
         | 3    | (x)_bg_(y)_fgFlip      | on BG Image (x) FG Image (y) is Flipped and overlaid         |
         | 4    | (x)_bg_(y)_fgFlip_mask | on BG Image (x) FG Image (y) is Flipped and overlaid with corresponding mask |
         
         
         
     - Folder Structure: 

         - We have created 10-Zip files as mentioned above with the image naming structure and distributed among the team.
         - 10-Zip files consists of the naming structure output_0.zip to output_9.zip (total of 800K Images)
         - Each zip file consists of 80K images
           - 20K BG_FG
           - 20K BG_FG_Flip
           - 20K BG_FG_mask
           - 20K BG_FGFlip_mask

     - Dense depthmap strategy
         - We took the images from the *.zip folder directly to create a depth maps, total of 400K Images..
         - The depth images are places at below places in respective team members google drive 

     1. Total images of each kind
        - BG-FG Images : 400k
        - BG-FG Mask Images : 400k
        - Dense Depth images for all BG-GF and BG-FG Flip : 400k
     2. The total size of the dataset
        
        - 1200K
     3. Mean/STD values for your fg_bg, masks and depth images
        
        - **fg_bg images**
        - Red Channel mean of bg_fg images 0.5360746
          - Green Channel mean of bg_fg images 0.46813193
        - Blue Channel mean of bg_fg images 0.3992631
          - Red Channel std dev of bg_fg images 0.25605327
        - Blue Channel std dev of bg_fg images 0.24445891
          - Green Channel std dev of bg_fg images 0.2392703
        - **fg_bg flip images**
          - Red Channel mean of bg_fg_flip images 0.5360829
          - Green Channel mean of bg_fg_flip images 0.46815184
          - Blue Channel mean of bg_fg_flip images 0.39928764
          - Red Channel std dev of bg_fg_flip images 0.25606075
          - Blue Channel std dev of bg_fg_flip images 0.24446595
          - Green Channel std dev of bg_fg_flip images 0.23927386
        - **bg_fg_mask images**
          - mean of bg_fg_mask images: 0.054658275
          - std dev of bg_fg_mask images: 0.22315167
        - **bg_fg_flipmask**
          - mean of bg_fg_mask images: 0.054657616
          - std dev of bg_fg_mask images: 0.2231518
        - **depth map images:**
          - Red Channel mean of depth_map images 0.82477474
          - Green Channel mean of depth_map images 0.61797315
          - Blue Channel mean of depth_map images 0.64635575 
          - Red Channel std dev of depth_map images 0.27929935 
          - Blue Channel std dev of depth_map images 0.3886217 
          - Green Channel std dev of depth_map images 0.3228312
        
     
  3. Show your dataset the way I have shown above in this readme
  
     1. Select "scene" images. Like the front of shops, etc. We call this background.
  
      <img src=".\assets\bg_10.png" alt="BG" style="zoom:150%;" />
  
   2. Find or make 100 images of objects with transparent background. Use GIMP. We call this foreground.
  
      <img src=".\assets\fg_10.png" alt="image-20200510222747744" style="zoom:150%;" />
  
   3. Create 100 Masks for the above image. Use GIMP
  
      <img src=".\assets\fg_mask_10.png" alt="image-20200510223035271" style="zoom:150%;" />
  
   4. Overlay the foreground on top or background randomly. Flip foreground as well. We call this fg_bg
  
      - Normal
  
        <img src=".\assets\fg_bg_10.png" alt="FG_BG" style="zoom:150%;" />
  
      - Flipped
  
        <img src=".\assets\fg_bg_10_flipped.png" alt="FG_BG_Flipped" style="zoom:150%;" />
  
   5. Don't forget to create equivalent masks for these images:
  
      - Normal
  
        <img src=".\assets\fg_bg_mask_overlay.png" alt="FG_BG_Overlay" style="zoom:150%;" />
  
      - Flipped
  
        <img src=".\assets\bg_fg_mask_flipped.png" alt="." style="zoom:150%;" />
  
   6. Use this or similar [Depth Models (Links to an external site.)](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) to create depth maps for the fg_bg images:
  
      <img src=".\assets\densedepth_10.png" alt="DenseDepth" style="zoom:150%;" />
  
4. Explain how you created your dataset
  
   1. how were fg created with transparency
        - Power point has been used for creating transparent Images
   2. how were masks created for fgs
        - Image Magic has been used to generate masks
   3. how did you overlay the fg over bg and created 20 variants
        - PIL module has been used to overlay and flip for 20-variants
     4. how did you create your depth images?
        - Used shared DenseDepth git hub repo
            - Selected BG,FG images 
            - Overlayed FG on BG
            - Generated dataset as mentioned above in 10-zip files
            - We put these zip files in google drive of 3-Team members (4+3+3=10)
            - Made few changes in the DenseDepth github repo files (test.py,utils.py,*.ipynb and layers.py)
            - Summary: 
              - Read Images from memory by accessing the zipfile object.
              - We didn't extract the zip into the disk.
              - We processed 10K images from the zipfile at a time in batches to avoid OOM and RAM issues.
              - Changes made like, provided arguments for the batchsize and dataset selection range
              - Made changes in utils.py w.r.t display_images and loading images.
              - Changes made to save the depthmap images directly to google drive with file names (corelated with dataset name like..(x)_bg_(y)_fg.jpg)
              - Without any issues we are able to process 5K images in 8-10 minutes

  

- **Submission**

  1. Share the link to the readme file for your Assignment 15A. Read the assignment again to make sure you do not miss any part which you need to explain. -2500
     - https://github.com/praveenraghuvanshi1512/EVA4/tree/master/Session-15/Assignment-15/Assignment-15-A

  2. Share the link to your notebook which you used to create this dataset. We are expecting to see how you manipulated images, overlay code, depth predictions. -250
     - Github: https://github.com/praveenraghuvanshi1512/EVA4/blob/master/Session-15/Assignment-15/Assignment-15-A/EVA_4_S15_A15_A_DenseDepth.ipynb
     - Colab: https://colab.research.google.com/drive/18LZZYmjUTxvxkuXSeZj_duevLTTZ-wMo?usp=sharing
  
  3. Surprise question. -Surprise Marks. 
     - Hopefully Rohan will bless us :-)