# EVA4 - Session 10 : 18-March-2020

#### Video  : https://youtu.be/MQiM-tF0now

[![EVA-4 Session 10](http://img.youtube.com/vi/MQiM-tF0now/0.jpg)](https://youtu.be/MQiM-tF0now)

### Links

- [Session-10 Pdf](S10.pdf)

- [Video](https://youtu.be/MQiM-tF0now)

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