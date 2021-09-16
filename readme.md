**First of all, I inform you that I made the code using the following.**  
: https://www.kaggle.com/bassbone/dog-vs-cat-transfer-learning-by-pytorch
##  CAM comparison among 4 famous models (EfficientNet-B0, EfficientNet-B1, VGG16, and ResNet50) on Cat2Dog dataset
Compare CAM results for cat2dog images for each model of EfficientNet-B0, EfficientNet-B1, VGG16, and ResNet50.  
Through this, we select the most efficient way to give Feature Importance.  
![image](https://user-images.githubusercontent.com/40943064/133630586-1bbb498f-63ae-4d06-a826-649ef1e73de2.png)  
![image](https://user-images.githubusercontent.com/40943064/133632417-909207e5-6350-41d5-9c2b-30ad214568d8.png)  

link : https://www.kaggle.com/c/dogs-vs-cats  
## EfficientNet
Indirect performance verification of EfficientNet is possible below.  
This is the classification performance obtained using the vast amount of data from ImageNet.  
![image](https://user-images.githubusercontent.com/40943064/133631365-1e1a1cd7-5b0d-437b-a1ef-b448ad91beb4.png)

## Quantitative model performance comparison
  
Since the total amount of learning is 5 epoch, the results cannot be considered with perfect accuracy.  
Furthermore the resolution is set to 128 and there is much room for accuracy to increase if the resolution is increased.  
VGG and ResNet used the Pre-trained model, while EfficientNet was trained from the beginning.


