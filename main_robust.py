from __future__ import print_function
#from IPython.core.debugger import set_trace
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
from torch.autograd import Variable
from itertools import starmap
import random
parser = argparse.ArgumentParser(description='Robust Neural Network Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
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
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=0)),
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
num_classes = 100

trainloader_noise = torch.utils.data.DataLoader(trainset_noise, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_noise = torch.utils.data.DataLoader(testset_noise, batch_size=100, shuffle=False, num_workers=2)
testloader_clean = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Return network & file name
def getNetwork(args):
    net = ResNet_2Read(50, num_classes)
    file_name = 'resnet-50'

    return net, file_name

# Test only option


if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'robust_readout_matching_moddedNormalize.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs, img_type="noise", compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Result (Noise Readout)\tAcc@1: %.2f%%" %(acc))

    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs, img_type="clean", compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    #variance = batch_var.mean()
    print("| Test Result (Clean Readout)\tAcc@1: %.2f%%" %(acc))

    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'_readout_matchrobust_SNRNormalized.t7')
    net = checkpoint['net']
    if len(checkpoint)>1:  
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['acc']
    else:
        best_acc = 100.0
        start_epoch = 200
else:
   # print('| Building net type [' + args.net_type + ']...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
w_loss = nn.MSELoss()
# Similarity Loss Computation

readout_losses = []
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, ((inputs1, targets1), (inputs2, targets2)) in enumerate(zip(trainloader_noise, trainloader_clean)):
 
        if use_cuda:
            inputs1, targets1 = inputs1.cuda(), targets1.cuda() # GPU settings
            inputs2, targets2 = inputs2.cuda(), targets2.cuda()
        optimizer.zero_grad()
        outputs_n = net(inputs1, img_type="noise", compute_similarity=False)
        l1 = criterion(outputs_n, targets1)
        l1.backward(retain_graph=True)
        #optimizer.step()
        
        #optimizer.zero_grad()
        outputs_c = net(inputs2, img_type="clean", compute_similarity=False)
        l2 = criterion(outputs_c, targets2)
        l2.backward(retain_graph=True)
        #optimizer.step()
        
        #optimizer.zero_grad()
        l3 = w_loss(outputs_n, outputs_c)
        readout_losses.append(l3.item())
        l3.backward(retain_graph=True)
        optimizer.step() # Optimizer update
        
        loss = l1 + l2 + l3
        train_loss += loss.item()
        _, predicted = torch.max(outputs_c.data, 1)
        total += targets2.size(0)
        correct += predicted.eq(targets2.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\t Total Loss: %.4f Acc@1: %.3f%%'
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
    for batch_idx, (inputs1, targets1) in enumerate(testloader_noise):
        if use_cuda:
            inputs1, targets1 = inputs1.cuda(), targets1.cuda()
            
        
        outputs_n = net(inputs1, img_type="noise", compute_similarity=False)
        loss = criterion(outputs_n, targets1)
        
        test_loss += loss.item()
        _, predicted1 = torch.max(outputs_n.data, 1)
        total1 += targets1.size(0)
        correct1 += predicted1.eq(targets1.data).cpu().sum()
    acc = 100.*correct1/total1


    for batch_idx, (inputs2, targets2) in enumerate(testloader_noise):
        if use_cuda:
            inputs2, targets2 = inputs2.cuda(), targets2.cuda()
            
        outputs_c = net(inputs2, img_type="clean", compute_similarity=False)
        loss2 = criterion(outputs_c, targets2)
        _, predicted2 = torch.max(outputs_c.data, 1)
        total2 += targets2.size(0)
        correct2 += predicted2.eq(targets2.data).cpu().sum()
    acc2 = 100.*correct2/total2
    print("\n| Validation Epoch #%d\t\t\tLoss (Noise): %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    print("\n| Validation Epoch #%d\t\t\tLoss (Clean): %.4f Acc@1: %.2f%%" %(epoch, loss2.item(), acc2))
    
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
        torch.save(state, save_point+file_name+'_robust.t7')
        best_acc = acc
    #accs.append(acc)
    #net.train()
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
np.save('epoch_readout_loss', np.asarray(readout_losses))
print('\n[Phase 4] : Testing model')
print('* Test results : Acc@1 = %.2f%%' %(best_acc))
print('| Saving model...')
state = {
    'net':net.module if use_cuda else net,
}
if not os.path.isdir('checkpoint'):
    os.mkdir('checkpoint')
save_point = './checkpoint/'+'cifar100'+os.sep
if not os.path.isdir(save_point):
    os.mkdir(save_point)
torch.save(state, save_point+file_name+'_readout_match_SNRNormalized.t7')
