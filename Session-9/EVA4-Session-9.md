# EVA4 - Session 9 : 11-March-2020

#### Video  : https://youtu.be/6SltB4vxGic

[![EVA-4 Session 9](http://img.youtube.com/vi/6SltB4vxGic/0.jpg)](https://youtu.be/6SltB4vxGic)

### Links

- [Session-8 Pdf](S9.pdf)

- [Video](https://youtu.be/6SltB4vxGic)

#### Assignment

- 

#### Notes

- We need to think which augmentation strategy to use. If my test dataset doesn't have horizontal or vertical flip, there is no point of using it.
- Random easing + cutout = Patch gaussian augmentation
- If augmentation is not used properly, there will be decrease in accuracy
- Most of the Class Activation Maps(CAM) have a requirement of GAP
- GradCAM doesn't have a requirement of GAP.
- GradCAM is the x-ray of your image
- Don't pick 1 pixel for GradCAM
- Atlease take 7x7 for GradCAM
- This is one of the reason why we don't go below 7x7
- Cutout can help remove biases 



#### Questions

- 

#### Further Study

- Gaussian Distribution

### References

- https://github.com/uday96/EVA4-TSAI/blob/master/S9/EVA4_S9_Quiz.ipynb
- https://github.com/Gaju27/eva4/blob/master/S9/models/quizDNN.py
- 
- [https://github.com/mounikaduddukuri/S9/blob/master/QUIZ/QUIZ9.ipynb (Links to an external site.)](https://github.com/mounikaduddukuri/S9/blob/master/QUIZ/QUIZ9.ipynb)
- [https://github.com/abhinavdayal/EVA4/tree/master/S9 (Links to an external site.)](https://github.com/abhinavdayal/EVA4/tree/master/S9)
- [https://github.com/gudaykiran/EVA4-Session-9/blob/master/Session9_Quiz9.ipynb (Links to an external site.)](https://github.com/gudaykiran/EVA4-Session-9/blob/master/Session9_Quiz9.ipynb)
- [https://github.com/Lakshman511/EVA4/tree/master/s9/Quiz (Links to an external site.)](https://github.com/Lakshman511/EVA4/tree/master/s9/Quiz)
- [https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/S9_Quiz.ipynb (Links to an external site.)](https://github.com/bharathts1507/TSAI-Assignments-EVA4/blob/master/S9_Quiz.ipynb)
- [https://github.com/DrVenkataRajeshKumar/S9/blob/master/QuizS9.ipynb (Links to an external site.)](https://github.com/DrVenkataRajeshKumar/S9/blob/master/QuizS9.ipynb)
- [https://github.com/uday96/EVA4-TSAI/blob/master/S9/EVA4_S9_Quiz.ipynb (Links to an external site.)](https://github.com/uday96/EVA4-TSAI/blob/master/S9/EVA4_S9_Quiz.ipynb)
- [https://github.com/srilakshmiv14/EVA4-Session-9/blob/master/S9_Q9.ipynb (Links to an external site.)](https://github.com/srilakshmiv14/EVA4-Session-9/blob/master/S9_Q9.ipynb)
- [https://colab.research.google.com/github/meenuraji/S9/blob/master/Copy_of_QuizS9.ipynb (Links to an external site.)](https://colab.research.google.com/github/meenuraji/S9/blob/master/Copy_of_QuizS9.ipynb)