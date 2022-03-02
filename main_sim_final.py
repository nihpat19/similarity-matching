from __future__ import print_function
from pdb import set_trace
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
import scipy.ndimage as ndimage
from networks import *
import random
parser = argparse.ArgumentParser(description='Training ResNet with Similarity Regularization')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--sim', '-s', action='store_true', help='Train and regularize using similarity matching')
parser.add_argument('--lamb', '-lb', default=1.1, type=float, help='base value for regularization strength (similarity only)')
parser.add_argument('--eta', '-e', default=0.2, type=float, help='exponential value for regularization strength (similarity only)')
parser.add_argument('--noise', '-n', action='store_true', help='Train using noise-augmented images')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()

# Hyper Parameter settings
sim_learning = args.sim
use_cuda = torch.cuda.is_available()
best_acc = 0
use_noise = args.noise

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
    transforms.RandomChoice(gaussian_transforms)
    #transforms.ToTensor()
])
    
transform_train_clean = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100'])
]) # meanstd transformation
    
transform_test_noise = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.RandomChoice(gaussian_transforms)
    #transforms.ToTensor()

])

transform_test_noise_0 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=0))
    #transforms.ToTensor()
])

transform_test_noise_1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=1))
    #transforms.ToTensor()
])

transform_test_noise_2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=2))
    #transforms.ToTensor()
])

transform_test_noise_5 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=5))
    #transforms.ToTensor()
])

transform_test_noise_10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100']),
    transforms.Lambda(lambda x:ndimage.gaussian_filter(x, sigma=10))
    #transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean['cifar100'], cf.std['cifar100'])
])

print("| Preparing CIFAR-100 dataset...")
sys.stdout.write("| ")
trainset_noise = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_noise)

trainset_clean = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train_clean)

testset_noise = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise)

testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
testset_noise0 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_0)
testset_noise1 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_1)
testset_noise2 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_2)
testset_noise5 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_5)
testset_noise10 = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test_noise_10)
num_classes = 100
    
    
    

trainloader_noise = torch.utils.data.DataLoader(trainset_noise, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader_clean = torch.utils.data.DataLoader(trainset_clean, batch_size=batch_size, shuffle=True, num_workers=2)
testloader_noise = torch.utils.data.DataLoader(testset_noise, batch_size=100, shuffle=False, num_workers=2)
testloader_clean = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_0 = torch.utils.data.DataLoader(testset_noise0, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_1 = torch.utils.data.DataLoader(testset_noise1, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_2 = torch.utils.data.DataLoader(testset_noise2, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_5 = torch.utils.data.DataLoader(testset_noise5, batch_size=100, shuffle=False, num_workers=2)
testloader_noise_10 = torch.utils.data.DataLoader(testset_noise10, batch_size=100, shuffle=False, num_workers=2)

# Return network & file name
def getNetwork(args):
    net = ResNet(50, num_classes)
    file_name = 'resnet-50'
    return net, file_name


# Test only option


if (args.testOnly):
    print('\n[Test Phase] : Model setup')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'_similarity_regularized_allLayers_layerwiseRegStrength_lambda20_eta0.02.t7')
    #set_trace()
    net = checkpoint['net']
    
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise_0):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Results (0 standard deviations) \tAcc@1: %.2f%%" %(acc))
    
    
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
    print("| Test Results (1 standard deviation) \tAcc@1: %.2f%%" %(acc))
    
    
    
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise_2):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Results (2 standard deviations) \tAcc@1: %.2f%%" %(acc))
    
    
    
    
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise_5):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Results (5 standard deviations) \tAcc@1: %.2f%%" %(acc))
    
    
    test_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(testloader_noise_10):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs, compute_similarity=False)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total
    print("| Test Results (10 standard deviations) \tAcc@1: %.2f%%" %(acc))
        
    sys.exit(0)

# Model
print('\n[Phase 2] : Model setup')
if args.resume:
    # Load checkpoint
    print('| Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: No checkpoint directory found!'
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+'cifar100'+os.sep+file_name+'_robust_unregularized_end.t7')
    net = checkpoint['net']
    if(len(checkpoint)>1):
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = 100
        start_epoch = 200
else:
    print('| Building net type [Resnet]...')
    net, file_name = getNetwork(args)
    net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=[0])
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()


if (sim_learning):
    print('| Loading Regularizer Network...')
    checkpoint_gauss = torch.load("./checkpoint/cifar100/resnet-50_readout_matchrobust_SNRNormalized.t7")
    robustNet = checkpoint_gauss['net']
    robustNet = torch.nn.DataParallel(robustNet, device_ids=[0])
# Similarity Loss Computation

def get_sim_loss(layer, matrix_n, matrix_r, eps, lamb = args.lamb, eta = args.eta):
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
    if(args.resume and start_epoch==200):
        params = net.module.linear.parameters()
    else:
        params = net.parameters()
    optimizer = optim.SGD(params, lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)
    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))
    if(use_noise):
        loader = trainloader_noise
    else:
        loader = trainloader_clean
    for batch_idx, (inputs_c, targets_c) in enumerate(loader):
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
    if sim_learning:
        testing = testloader_noise_1
    else:
        testing = testloader_clean
        
    for batch_idx, (inputs_n, targets_n) in enumerate(testing):
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
        if(sim_learning):
            torch.save(state, save_point+file_name+'_similarity_regularized_layerwiseRegStrength_lambda'+str(args.lamb)+'_eta'+str(args.eta)+'_SNRsimilarity.t7')
            #torch.save(state, save_point+file_name+'_similarity_regularized_10thLayer_noise.t7')

        else:
            torch.save(state, save_point+file_name+'_robust_unregularized.t7')
        best_acc = acc
if sim_learning:
    print('\n[Phase 3] : Training and Regularizing model')
else:
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
if(sim_learning):
   torch.save(state, save_point+file_name+'_similarity_regularized_layerwiseRegStrength_lambda'+str(args.lamb)+'_eta'+str(args.eta)+'_SNRsimiliarity_end.t7')
    #torch.save(state, save_point+file_name+'_similarity_regularized_10thLayer_noise_end.t7')

else:
    torch.save(state, save_point+file_name+'_unregularized_end.t7')
#np.save('similarity losses', np.array(sim_losses))
