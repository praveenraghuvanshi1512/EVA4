import os
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
def saveimage(images, outputdirectory, imagename):
    os.makedirs(outputdirectory, exist_ok=True)
    outputname = imagename
    outputpath = os.path.join(outputdirectory, outputname)
    save_image(images, outputpath)
    pilimg = PIL.Image.open(outputpath)
    return pilimg

def plotimages(device, classes, dataloader, numofimages=20):
    counter = 0
    fig = plt.figure(figsize=(10,9))
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        print(len(target))
        for index, label in enumerate(target):
          print(index)
          print(label)
          title = "{}".format(classes[label.item()])
          #print(title)
          ax = fig.add_subplot(4, 5, counter+1, xticks=[], yticks=[])
          #ax.axis('off')
          ax.set_title(title)
          #plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
          imshow(data[index].cpu())
          
          counter += 1
          if(counter==numofimages):
            break
        if(counter==numofimages):
            break
    return

def plotmetrics(trainaccuracies, testaccuracies, trainlosses, testlosses, savefilename):
    print(len(trainaccuracies))
    print(len(testaccuracies))
    
    fig, axs = plt.subplots(1, 2, figsize=(15,10))
    
    # Plot Accuracy
    axs[0].plot(trainaccuracies, label='Train Accuracy')
    axs[0].plot(testaccuracies, label='Test Accuracy')
    axs[0].set_title("Accuracy")
    axs[0].legend(loc="upper left")
    
    # Plot loss
    axs[1].plot(trainlosses, label='Train Loss')
    axs[1].plot(testlosses, label='Test Loss')
    axs[1].set_title("Loss")
    axs[1].legend(loc="upper left")
    
    plt.show()
    fig.savefig("{}.png".format(savefilename))
    
def plotmisclassifiedimages(model, device, classes, testloader, numofimages = 25, savefilename="misclassified"):
    model.eval()
    misclassifiedcounter = 0
    fig = plt.figure(figsize=(10,9))
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
        pred_marker = pred.eq(target.view_as(pred))   
        wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
        for idx in wrong_idx:
          index = idx[0].item()
          title = "True:{},\n Pred:{}".format(classes[target[index].item()], classes[pred[index][0].item()])
          #print(title)
          ax = fig.add_subplot(5, 5, misclassifiedcounter+1, xticks=[], yticks=[])
          #ax.axis('off')
          ax.set_title(title)
          #plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
          imshow(data[index].cpu())
          
          misclassifiedcounter += 1
          if(misclassifiedcounter==numofimages):
            break
        
        if(misclassifiedcounter==numofimages):
            break
        
    fig.tight_layout()
    fig.savefig("{}.png".format(savefilename))
    return

def savemisclassifiedimages(model, device, classes, testloader, outputdirectory, numofimages = 25):
    model.eval()
    os.makedirs(outputdirectory, exist_ok=True)
    
    misclassifiedimagenames = []
    misclassifiedtitles = []
    misclassifiedcounter = 0
    fig = plt.figure(figsize=(10,9))
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)     # get the index of the max log-probability
        pred_marker = pred.eq(target.view_as(pred))   
        wrong_idx = (pred_marker == False).nonzero()  # get indices for wrong predictions
        for idx in wrong_idx:
          index = idx[0].item()
          title = "True:{},\n Pred:{}".format(classes[target[index].item()], classes[pred[index][0].item()])
          misclassifiedtitles.append(title)
          # print(title)
          ax = fig.add_subplot(5, 5, misclassifiedcounter+1, xticks=[], yticks=[])
          # ax.axis('off')
          ax.set_title(title)
          # plt.imshow(data[index].cpu().numpy().squeeze(), cmap='gray_r')
          imshow(data[index].cpu())

          outputname = "{}.jpg".format(str(misclassifiedcounter))
          misclassifiedimagenames.append(outputname)
          outputpath = os.path.join(outputdirectory, outputname)

          save_image(data[index].cpu(), outputpath)
          
          misclassifiedcounter += 1
          if(misclassifiedcounter==numofimages):
            break
        
        if(misclassifiedcounter==numofimages):
            break
        
    fig.tight_layout()
    fig.savefig("misclassified.png")
    return misclassifiedimagenames, misclassifiedtitles
    
def plotmisclassifiedgradcamimages(misclassifiedgradcamimages, titles, savefilename):
    to_pil = transforms.ToPILImage()
    fig,ax = plt.subplots(5,5,figsize=(30,10))
    fig.suptitle('Misclassified Gradcam Images')
    imageindex = 0
    
    for i in range(5):
      for j in range(5):
        image = to_pil(misclassifiedgradcamimages[imageindex])
        ax[i][j].imshow(image)
        ax[i][j].axis('off')
        ax[i][j].set_title(titles[imageindex])
        imageindex = imageindex+1
    
    fig.tight_layout()
    fig.savefig("{}.png".format(savefilename))