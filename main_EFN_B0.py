#%% import library
import pandas as pd
import os, glob
import torch
from dataset import *
from CAM.cam import GradCAM
import torch.nn.functional as F
from CAM.visualize import visualize, reverse_normalize
import matplotlib.pyplot as plt
import torch.utils.data as data

# from PIL import Image
# import numpy as np
# from torchvision.transforms.transforms import ToTensor

# from statistics import mean
# os.chdir('D:/공부/Study_EfficientNet/')
SIZE = 240
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DATA_DIR = 'D:/공부/dataset/cat2dog'
MAIN_DIR = 'D:/공부/Study_EfficientNet'
TRAIN_DIR_A = DATA_DIR + '/trainA'
TRAIN_DIR_B = DATA_DIR + '/trainB'
TEST_DIR = DATA_DIR + '/test'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(MAIN_DIR + '/rst', exist_ok=True)
train_list     = pd.Series(glob.glob(os.path.join(TRAIN_DIR_A, '*.jpg')) + glob.glob(os.path.join(TRAIN_DIR_B, '*.jpg')))
test_list      = pd.Series(glob.glob(os.path.join(TEST_DIR, '*.jpg')))

train_dataset = DogVsCatDataset(train_list, transform=ImageTransform(SIZE, MEAN, STD), phase='val')
train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
#%% 
net, acc, loss, time = f_train(train_list, 'efficientnet-b0')
net.eval()
target_layer = net._conv_head
target_layer.requires_grad_(True)
net_cam = GradCAM(net, target_layer) # load gpu!?
#%% Create CAM images using trained model

# %matplotlib qt5
plot_no = 0
rst_no = 0

plt.figure(figsize=(15,8))
for inputs, labels in train_dataloader:
    plot_no +=1

    CAM_rst, _ = net_cam(inputs.to(DEVICE), labels[0])
    CAM_resize = F.interpolate(CAM_rst, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
    img        = reverse_normalize(inputs)
    heatmap, _ = visualize(img, CAM_rst)

    plt.subplot(4,8, plot_no)
    plt.imshow(heatmap[0].swapaxes(0,1).swapaxes(1,2))
    plt.axis('off')
    if plot_no == 32:
        print(MAIN_DIR + f'/rst/CAM_{rst_no}.png')
        plot_no = 0
        rst_no+=1
        plt.savefig(MAIN_DIR + f'/rst/CAM_{rst_no}.png')
        plt.clf()
#%%
torch.save(net.state_dict(), MAIN_DIR + f'/rst/CAM_EFN_B0.pt')