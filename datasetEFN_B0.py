import torch
import time, copy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import models
import os
SIZE = 128
NUM_CLASS = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 16
LR = 1e-4
DATA_DIR = 'D:/공부/dataset/cat2dog'
TRAIN_DIR = DATA_DIR + '/train'
TEST_DIR = DATA_DIR + '/test'
TEST_RATIO = .1
NUM_EPOCH = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SEED = 2021

#%%

# def load(net_cam, MAIN_DIR):
#     params = torch.load(os.path.join(MAIN_DIR + f'/rst/CAM_EFN_B0.pt'))
#     net_cam.load_sdastate_dict(params)


## class definition
class ImageTransform():
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }
    def __call__(self, img, phase):
        return self.data_transform[phase](img)

class DogVsCatDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):    
        
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img, self.phase)

        if 'cat.' in img_path:
            label = 0
        elif 'dog.' in img_path:
            label = 1

        return img_transformed, label

## function
def create_params_to_update(net, update_params_name_1):
    params_to_update_1 = []
    for name, param in net.named_parameters():
        if name in update_params_name_1:
            param.requires_grad = True
            params_to_update_1.append(param)
            #print("{} 1".format(name))
        else:
            param.requires_grad = False
            #print(name)

    params_to_update = [
        {'params': params_to_update_1, 'lr': LR}
    ]
    
    return params_to_update

def adjust_learning_rate(optimizer, epoch):
    lr = LR * (0.1**(epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def train_model(net, train_dataloader, criterion, optimizer, num_epoch):
    
    since = time.time()
    # best_model_wts = copy.deepcopy(net.state_dict())
    best_acc, best_loss = 0.0, float('inf')
    net = net.to(DEVICE)
    
    for epoch in range(num_epoch):
        print('Epoch {}/{} ---'.format(epoch + 1, num_epoch))
        
        EPOCH_ITER = 0
        net.train()
            
        epoch_loss, epoch_corrects = 0.0, 0
        
        for inputs, labels in train_dataloader:
            EPOCH_ITER += BATCH_SIZE

            
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(mode = True):
                outputs = net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * inputs.size(0)
                
                epoch_corrects += torch.sum(preds == labels.data).item()
                

                if EPOCH_ITER % (BATCH_SIZE * 10) == 0:    # print training losses and save logging information to the disk
                    print('STEP: ' + str(EPOCH_ITER) + ' of ' + str(len(train_dataloader.dataset)) + ' / Accuracy : ' + str(round(100 * epoch_corrects / EPOCH_ITER, 1)) + '%')
        epoch_loss = epoch_loss / len(train_dataloader.dataset)
        epoch_acc = float(epoch_corrects) / len(train_dataloader.dataset)
        
        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # net.load_state_dict(best_model_wts)
    return net, best_acc, best_loss, time_elapsed

    
def f_train(train_list, MODEL_NAME):
    train_dataset = DogVsCatDataset(train_list, transform=ImageTransform(SIZE, MEAN, STD), phase='train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

    
    print('Current Train Model : ' + MODEL_NAME)


    if MODEL_NAME in ['efficientnet-b0', 'efficientnet-b1'] : # 20MB, 
        net = EfficientNet.from_pretrained(MODEL_NAME)
        net._fc.out_features = NUM_CLASS
        update_params_name_1 = ['_fc.weight', '_fc.bias', '_conv_head.weight', '_conv_head.bias']
        

    elif MODEL_NAME == 'VGG16': # 524MB
        net = models.vgg16(pretrained=True)
        net.classifier[6].out_features = NUM_CLASS
        update_params_name_1 = ['classifier.0.weight','classifier.0.bias','classifier.3.weight','classifier.3.bias','classifier.6.weight','classifier.6.bias']

    elif MODEL_NAME == 'RESNET50': # 98MB
        net = models.resnet50(pretrained=True)
        net.fc.out_features = NUM_CLASS
        update_params_name_1 = ['fc.weight', 'fc.bias','layer3.2.conv1.weight','layer3.2.conv1.bias','layer3.2.conv2.weight','layer3.2.conv2.bias','layer3.2.conv3.weight','layer3.2.conv3.bias']
        

    

    params_to_update = create_params_to_update(net, update_params_name_1)
    optimizer = optim.Adam(params=params_to_update, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()

    net, acc, loss, time_elapsed = train_model(net, train_dataloader, criterion, optimizer, NUM_EPOCH)
    #print(acc)
    #print(loss)
    return net, acc, loss, 1000*time_elapsed/(NUM_EPOCH*25000)