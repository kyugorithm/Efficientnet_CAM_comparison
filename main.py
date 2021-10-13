#%% import library
import matplotlib
from matplotlib.pyplot import imshow
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
import numpy as np

# from statistics import mean
# os.chdir('D:/공부/Study_EfficientNet/')
SIZE = 240
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

approach_root = 1 if  'gdrive' in os.getcwd().split('/') else 2

if approach_root == 1 :
    DATA_DIR = '/content/gdrive/MyDrive/UGATIT/dataset/cat2dog'
elif approach_root == 2:
    DATA_DIR = 'D:/공부/dataset/cat2dog'

TRAIN_DIR_A = DATA_DIR + '/trainA'
TRAIN_DIR_B = DATA_DIR + '/trainB'
TEST_DIR = DATA_DIR + '/test'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR + '/rst', exist_ok=True)
train_list     = pd.Series(glob.glob(os.path.join(TRAIN_DIR_A, '*.jpg')) + glob.glob(os.path.join(TRAIN_DIR_B, '*.jpg')))
test_list      = pd.Series(glob.glob(os.path.join(TEST_DIR, '*.jpg')))

train_dataset = DogVsCatDataset(train_list[:50], transform=ImageTransform(SIZE, MEAN, STD), phase='val')
train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

#%%
# , 
MODEL_NAMES = ['efficientnet-b0', 'efficientnet-b1','VGG16', 'RESNET50']

NETS = {}
ACCS = {}
LOSSES = {}
TARGET_LAYER = {}
NET_CAM = {}
TIMES = {}
# print(30*'#')
# print(train_list[0])
# print(30*'#')

for MODEL_NAME in MODEL_NAMES:
    NETS[MODEL_NAME], ACCS[MODEL_NAME], LOSSES[MODEL_NAME], TIMES[MODEL_NAME] = f_train(train_list, MODEL_NAME)
    NETS[MODEL_NAME].eval()

    if MODEL_NAME in ['efficientnet-b0', 'efficientnet-b1'] : # ~ 30MB
        TARGET_LAYER[MODEL_NAME] = NETS[MODEL_NAME]._conv_head
    elif MODEL_NAME == 'VGG16': # 524MB
        TARGET_LAYER[MODEL_NAME] = NETS[MODEL_NAME].features[28]
    elif MODEL_NAME == 'RESNET50': # 98MB
        TARGET_LAYER[MODEL_NAME] = NETS[MODEL_NAME].layer4[2].conv3
        
    TARGET_LAYER[MODEL_NAME].requires_grad_(True)
    NET_CAM[MODEL_NAME] = GradCAM(NETS[MODEL_NAME], TARGET_LAYER[MODEL_NAME]) # load gpu!?

#%% prediction

cnt = 0
raw_no = 0
# %matplotlib qt5
plt.figure(figsize=(16,8))
for inputs, labels in train_dataloader:
    cnt += 1
    # heatmap
    for i, MODEL_NAME in enumerate(MODEL_NAMES):
        CAM_rst, _ = NET_CAM[MODEL_NAME](inputs.to(DEVICE), labels[0])
        CAM_resize = F.interpolate(CAM_rst, size=(SIZE, SIZE), mode='bilinear', align_corners=False)
        img = reverse_normalize(inputs)

        heatmap, _ = visualize(img, CAM_rst)
        plot_no = i+1 + 4 * raw_no
        plt.subplot(5, 4, plot_no)
        plt.imshow(heatmap[0].swapaxes(0,1).swapaxes(1,2))
        plt.axis('off')
        if i == 3:
            raw_no += 1

        if raw_no == 5:
            raw_no = 0
    if plot_no == 20:
        plt.savefig(DATA_DIR + '/rst/' + str(cnt-19) + '_' + str(cnt) + '.png')
        plt.clf()

# print(ACCS)
# print(LOSSES)
# print(TIMES)
