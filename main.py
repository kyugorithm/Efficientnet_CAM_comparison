#%% import library
import matplotlib
from matplotlib.pyplot import imshow
import os 
import pandas as pd
import os, glob
from PIL import Image
import torch
from torchvision.transforms.transforms import ToTensor
from dataset import *
from CAM.cam import GradCAM, CAM
import torch.nn.functional as F
from CAM.visualize import visualize, reverse_normalize
import matplotlib.pyplot as plt
from torchvision.utils import save_image


# from statistics import mean
# os.chdir('D:/공부/Study_EfficientNet/')
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# DATA_DIR = 'D:/공부/dataset/cat2dog'
DATA_DIR = '/content/gdrive/MyDrive/UGATIT/dataset/cat2dog'
TRAIN_DIR_A = DATA_DIR + '/trainA'
TRAIN_DIR_B = DATA_DIR + '/trainB'
TEST_DIR = DATA_DIR + '/test'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('rst', exist_ok=True)
train_list     = pd.Series(glob.glob(os.path.join(TRAIN_DIR_A, '*.jpg')) + glob.glob(os.path.join(TRAIN_DIR_B, '*.jpg')))
test_list      = pd.Series(glob.glob(os.path.join(TEST_DIR, '*.jpg')))
#%%
# 'efficientnet-b0', 'efficientnet-b1', 
MODEL_NAMES = ['VGG16', 'RESNET50']

NETS = {}
ACCS = {}
LOSSES = {}

for MODEL_NAME in MODEL_NAMES:
    NETS[MODEL_NAME], ACCS[MODEL_NAME], LOSSES[MODEL_NAME] = f_train(train_list, MODEL_NAME)
    NETS[MODEL_NAME].eval()

# target_layer = net._conv_head
# target_layer.requires_grad_(True)
# netCAM = GradCAM(net, target_layer) # load gpu!?
# #%% prediction
# train_dataset = DogVsCatDataset(train_list, transform=ImageTransform(SIZE, MEAN, STD), phase='val')
# train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
# cnt = 0
# # %matplotlib qt5
# # plt.figure(figsize=(16,8))
# for inputs, labels in train_dataloader:
#     cnt += 1
    
#     CAM_rst, _ = netCAM(inputs.to(DEVICE), labels[0])
#     CAM_resize = F.interpolate(CAM_rst, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
#     img = reverse_normalize(inputs)
    
#     heatmap, _ = visualize(img, CAM_rst)
#     save_image(heatmap, './rst/' + train_list[cnt].split('\\')[-1] + '.png')
# %%
