#%% Load Model
import torch
from efficientnet_pytorch import EfficientNet
from CAM.cam import GradCAM
import os 
#%% 0) Basic options
PT_DIR    = 'D:/공부/Study_EfficientNet/rst/CAM_EFN_B0.pt'
DEVICE    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASS = 2

# 1) Load : pretrained EFN-B0
net = EfficientNet.from_pretrained('efficientnet-b0')
# 2) FIX : convert class number for your data
net._fc.out_features = NUM_CLASS
net.to(DEVICE)
# 3) Load : pretrained network
net.load_state_dict(torch.load(PT_DIR))
net.eval()
# 3) get target layer and make grad. cam module
target_layer = net._conv_head
target_layer.requires_grad_(True)
net_cam = GradCAM(net, target_layer)