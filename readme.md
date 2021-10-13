**First of all, I inform you that I made the code using the following.**  
The performance of EfficientNet should be similar, but VGG or ResNet appeared to be superior.  
However, in fact, the accuracy verified by someone is more than 98% based on B0. If you find a problem, please suggest an issue.

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
<img src="https://user-images.githubusercontent.com/40943064/133789834-beb2bf4c-c0bb-4e73-b81e-ae79c4485ce3.png" width = 600 align="center">  
  1. EN:B0(96.1%)  
  2. EN:B1(%)  
  3. VGG16(96.0%)  
  4. RES50(%)  
![image](https://user-images.githubusercontent.com/40943064/133790845-cee59c57-8f45-4c8a-9ef6-c376cc35f022.png)  
![image](https://user-images.githubusercontent.com/40943064/133791205-0fa3f9a8-dddf-4dc2-97d0-c90df05e37c4.png)  
![image](https://user-images.githubusercontent.com/40943064/133791580-8eb6e056-8e6e-4788-80f0-f796cbb6e203.png)  
