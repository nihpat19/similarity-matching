from __future__ import print_function
from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np
import torchvision
import torchvision.transforms as transforms
#import ipdb
import os
import sys
import time
import argparse
import datetime
import scipy.ndimage as ndimage
from networks import *
import random
parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
sim_learning = True
use_cuda = torch.cuda.is_available()
best_acc = 0

start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
torch.manual_seed(2809)
gaussian_transforms = [
    transforms.Lambda(lambda x: ndimage.gaussian_filter(x, sigma=0)),
    transforms.Lambda(lambda x: ndimage.gaussian_filter(x, sigma=1)),
    transforms.Lambda(lambda x: ndimage.gaussian_filter(x, sigma=2)),
    transforms.Lambda(lambda x: ndimage.gaussian_filter(x, sigma=5)),
    transforms.Lambda(lambda x: ndimage.gaussian_filter(x, sigma=10))
    ]
transform_train_noise = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.RandomChoice(gaussian_transforms),
])
    
transform_train_clean = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
]) # meanstd transformation
    
transform_test_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.RandomChoice(gaussian_transforms),
])

transform_test_noise_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=1)),

])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
])

print("| Preparing CIFAR-100 dataset...")
sys.stdout.write("| ")
trainset_noise = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_noise)

trainset_clean = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_clean)

testset_noise = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
testset_noise1 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_1)
num_classes = 100
    
    
    

trainloader_noise = torch.utils.data.DataLoader(trainset_noise, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_noise = torch.utils.data.DataLoader(testset_noise, batch_size=100, shuffle=False, num_workers=2)
testloader_clean = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_1 = torch.utils.data.DataLoader(testset_noise1, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    net = ResNet(args.depth, num_classes)
    file_name = 'resnet-'+str(args.depth)
    return net, file_name
if (sim_learning):
    checkpoint_gauss = torch.load("./checkpoint/cifar100/resnet-50readout_matching_properTransform.t7")
    robustNet = checkpoint_gauss['net']
    robustNet = torch.nn.DataParallel(robustNet, device_ids=[0])

# Test only option


if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'_similarity_regularized_layerwiseRegStrength_lambda40.0_eta0.01_end.t7')

    net = checkpoint['net']
    
    if use_cuda:
        net.cuda()
        #net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise_1):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result\tAcc@1: %.2f%%" %(acc))
    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'_similarity_regularized_layerwiseRegStrength_lambda20_eta0.002.t7')
    net = checkpoint['net']
    if(len(checkpoint)>1):
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = 100
        start_epoch = 200
else:
    print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

# Similarity Loss Computation

def get_sim_loss(layer, matrix_n, matrix_r, eps, lamb = 20, eta = 0.02):
    reg_strength = lamb**(1+layer*eta)
    mn = (1-eps)*matrix_n
    mr= (1-eps)*matrix_r
    loss = ((0.5*torch.log((1+mn)/(1-mn)))- (0.5*torch.log((1+mr)/(1-mr))))**2
    if torch.isnan(loss.mean()):
        set_trace()
    return reg_strength*loss.mean()
# Training
sim_losses = []
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs_c, targets_c) in enumerate(trainloader_clean):
        if use_cuda:
              inputs_c, targets_c = inputs_c.cuda(), targets_c.cuda()
        optimizer.zero_grad()
        if(sim_learning):
            (outputs, matrices_reg) = net(inputs_c, compute_similarity=True)
            (_, matrices_rob) = robustNet(inputs_c, img_type="clean", compute_similarity=True)
            
            loss_similarity = 0.
            for i, (r, g) in enumerate(zip(matrices_reg, matrices_rob)):
                
                sim_loss = get_sim_loss(i, r,g, 1e-4)
                loss_similarity= loss_similarity + sim_loss
            loss = criterion(outputs, targets_c) + loss_similarity  # Loss
        else:
            outputs = net(inputs_c, compute_similarity=False)
            loss = criterion(outputs, targets_c)
            
        loss.backward()
        optimizer.step() # Optimizer update
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets_c.size(0)
        correct += predicted.eq(targets_c.data).cpu().sum()
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Loss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    (len(trainset_noise)//batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    for batch_idx, (inputs_n, targets_n) in enumerate(testloader_noise_1):
        if use_cuda:
            inputs_n, targets_n = inputs_n.cuda(), targets_n.cuda()
            
        
        outputs_n = net(inputs_n, compute_similarity=False)
        loss = criterion(outputs_n, targets_n)
        
        test_loss += loss.item()
        _, predicted1 = torch.max(outputs_n.data, 1)
        total1 += targets_n.size(0)
        correct1 += predicted1.eq(targets_n.data).cpu().sum()
    acc = 100.*correct1/total1


    print("\n| Validation Epoch #%d\t\t\tLoss (Noise): %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    
    # Save checkpoint when best model
    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
        state = {
                'net':net.module if use_cuda else net,
                'acc':acc,
                'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/'+'cifar100'+os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point+file_name+'_similarity_regularized_layerwiseRegStrength_lambda20_eta0.02_properTransform.t7')
        best_acc = acc
print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))
elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
print('| Saving model...')
state = {
    'net':net.module if use_cuda else net
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_point = './checkpoint/'+'cifar100'+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)
torch.save(state, save_point+file_name+'_similarity_regularized_layerwiseRegStrength_lambda20_eta0.02_properTransform_end.t7')
np.save('similarity losses', np.array(sim_losses))
