import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import datasets
import torch.backends.cudnn as cudnn
import os
import argparse

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR
import torch.nn.parallel
from utils import progress_bar

ImSize = 224

#cat2dog
data_dir = './data/cat2dog'
save_path = './checkpoint/ckpt_vgg_cat2dog.pth'
num_class = 2

#Animal10
# data_dir = './data/animal10'
#save_path = './checkpoint/ckpt_vgg_animal10.pth'
# num_class = 10

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_transforms = transforms.Compose([	
                                       transforms.RandomSizedCrop(ImSize),
									   transforms.RandomHorizontalFlip(),				   
									   transforms.ToTensor(),
									   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                       ])
test_transforms = transforms.Compose([transforms.Scale(256),
                                    transforms.CenterCrop(ImSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=test_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=8, shuffle=True)

net = models.vgg16(pretrained=True)
#net = models.resnet18(pretrained=True)

# change the number of classes 
#VGG
net.classifier[6] = nn.Linear(net.classifier[6].in_features, num_class)

#resnet18
#net.fc = nn.Linear(net.fc.in_features, num_class)

net = torch.nn.DataParallel(net).cuda()
net.to(device)

print(net)

# freeze convolution weights
# for param in net.features.parameters():
#     param.requires_grad = False

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
# optimizer
#optimizer = optim.SGD(net.classifier.parameters(), lr=args.lr, momentum=0.9)
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
# loss function
criterion = nn.CrossEntropyLoss()
scheduler = MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, save_path)
        best_acc = acc
    
    return test_loss/(batch_idx+1)


for epoch in range(start_epoch, start_epoch+100):
    #print('LR:', scheduler.get_lr())
    train(epoch)
    loss = test(epoch)
    scheduler.step(loss)
    