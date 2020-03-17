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

- 