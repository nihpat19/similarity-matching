import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.autograd import Variable
import sys
import torch.nn.init as init
import numpy as np
from IPython.core.debugger import set_trace

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2,2,2,2]),
        '34': (BasicBlock, [3,4,6,3]),
        '50': (Bottleneck, [3,4,6,3]),
        '101':(Bottleneck, [3,4,23,3]),
        '152':(Bottleneck, [3,8,36,3]),
    }

    return cf_dict[str(depth)]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out
    
    def get_similarity_matrix(self, x): #Computes batchwise correlation matrix for the feature activations of imagebatch x
        x = x.view(x.shape[0],-1)
        x_s = (x-x.mean(dim=1, keepdim=True))/x.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/x.shape[1]
        return sim
    
    def forward(self, x):
        (x, S)=x
        out = self.conv1(x)
        s1_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        s2_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn2(out))
        if(len(self.shortcut)>0):
            out2 = self.shortcut[0](x)
            sSkip = self.get_similarity_matrix(out2)
            out2 = self.shortcut[1](out2)
            out+=out2
            out = F.relu(out) 
            return (out, S+[s1_x, s2_x, sSkip])
        else:
            out += self.shortcut(x)
            out = F.relu(out) 
            return (out, S+[s1_x, s2_x])

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def get_similarity_matrix(self, x): #Compute correlation similarity matrix over image batch
        x = x.view(x.shape[0],-1)
#         x_f = x.detach().cpu().numpy()
#         xm = x_f.mean(axis=1, keepdims=True)
#         xsd = x_f.std(axis=1, keepdims=True)
        snr = x.mean(dim=1, keepdim=True)/x.std(dim=1, keepdim=True)
        x = snr*x
        x_s = (x-x.mean(dim=1, keepdim=True))/x.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/x.shape[1]
        return sim
    
    def forward(self, x):
        
### For Regularization Every 10th Layer
#         out = self.conv1(x)
#         out = F.relu(self.bn1(out))
#         out = self.conv2(out)
#         out = F.relu(self.bn2(out))
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out+=self.shortcut(x)
#         out = F.relu(out)
#         return out


### For Regularization Every Layer
        (x, S)=x
        out = self.conv1(x)
        s1_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        s2_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        s3_x = self.get_similarity_matrix(out)
        out = self.bn3(out)
        if(len(self.shortcut)>0):
            out2 = self.shortcut[0](x)
            sSkip = self.get_similarity_matrix(out2)
            out2 = self.shortcut[1](out2)
            out+=out2
            out = F.relu(out) 
            return (out, S+[s1_x, s2_x, s3_x, sSkip])
        else:
            out += self.shortcut(x)
            out = F.relu(out) 
            return (out, S+[s1_x, s2_x, s3_x])  

class ResNet(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def get_similarity_matrix(self, x):
        x = x.view(x.shape[0],-1)
        snr = x.mean(dim=1, keepdim=True)/x.std(dim=1, keepdim=True)
        x = snr*x
        x_s = (x-x.mean(dim=1, keepdim=True))/x.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/x.shape[1]
        return sim
        
        return sim
    
    def forward(self, x, compute_similarity=False):
        S = []
        out = self.conv1(x)
        S.append(self.get_similarity_matrix(out))
        out = F.relu(self.bn1(out))
        if compute_similarity:
            out, S = self.layer1((out, S))
            out, S = self.layer2((out, S))
            out, S = self.layer3((out, S))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            
            return (out, torch.stack(S))
        else:
            out, _ = self.layer1((out, S))
            out, _ = self.layer2((out, S))
            out, _ = self.layer3((out, S))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            return out
        

class ResNet_2Read(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet_2Read, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear_noise = nn.Linear(64*block.expansion, num_classes)
        self.linear_clean = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def get_similarity_matrix(self, x):
        x = x.view(x.shape[0],-1)
        snr = x.mean(dim=1, keepdim=True)/x.std(dim=1, keepdim=True)
        x = snr*x
        x_s = (x-x.mean(dim=1, keepdim=True))/x.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/x.shape[1]
        return sim
    
    def forward(self, x, img_type="noise", compute_similarity=False):
        S = []
        out = self.conv1(x)
        S.append(self.get_similarity_matrix(out))
        out = F.relu(self.bn1(out))
        if(compute_similarity):
### For Regularization at Every 10th Layer (with Skip Connections)        
# #             out_s1 = out
# #             out = F.relu(self.layer1[0].bn1(self.layer1[0].conv1(out)))
# #             out = F.relu(self.layer1[0].bn2(self.layer1[0].conv2(out)))
# #             out = self.layer1[0].bn3(self.layer1[0].conv3(out))
# #             out_s1 = self.layer1[0].shortcut[0](out_s1)
# # #             sSkip1 = self.get_similarity_matrix(out_s1)
# #             out_s1 = self.layer1[0].shortcut[1](out_s1)
# #             out+=out_s1
# #             out = F.relu(out)
                         
                         
#             out = self.layer1[:2](out)
#             out_1 = out
#             out = F.relu(self.layer1[2].bn1(self.layer1[2].conv1(out)))
#             out = F.relu(self.layer1[2].bn2(self.layer1[2].conv2(out)))
#             out = self.layer1[2].conv3(out)
#             s10_x = self.get_similarity_matrix(out)
#             out = self.layer1[2].bn3(out)
#             out+=self.layer1[2].shortcut(out_1)
#             out = F.relu(out)
            
# #             out_s2 = out
# #             out = F.relu(self.layer2[0].bn1(self.layer2[0].conv1(out)))
# #             out = F.relu(self.layer2[0].bn2(self.layer2[0].conv2(out)))
# #             out = self.layer2[0].bn3(self.layer2[0].conv3(out))
# #             out_s2 = self.layer2[0].shortcut[0](out_s2)
# # #             sSkip2 = self.get_similarity_matrix(out_s2)
# #             out_s2 = self.layer2[0].shortcut[1](out_s2)
# #             out+=out_s2
# #             out = F.relu(out)           
                         
            
#             out = self.layer2[:3](out)
#             out_2 = out
#             out = self.layer2[3].conv1(out)
#             s20_x = self.get_similarity_matrix(out)
#             out = F.relu(self.layer2[3].bn1(out))
#             out = self.layer2[3].conv2(out)
# #             #s21_x = self.get_similarity_matrix(out)
#             out = F.relu(self.layer2[3].bn2(out))
#             out = self.layer2[3].bn3(self.layer2[3].conv3(out))
#             out += self.layer2[3].shortcut(out_2)
#             out = F.relu(out)

# #             out_s3 = out
# #             out = F.relu(self.layer3[0].bn1(self.layer3[0].conv1(out)))
# #             out = F.relu(self.layer3[0].bn2(self.layer3[0].conv2(out)))
# #             out = self.layer3[0].bn3(self.layer3[0].conv3(out))
# #             out_s3 = self.layer3[0].shortcut[0](out_s3)
# # #             sSkip3 = self.get_similarity_matrix(out_s3)
# #             out_s3 = self.layer3[0].shortcut[1](out_s3)
# #             out+=out_s3
# #             out = F.relu(out)
                         
#             out = self.layer3[:2](out)
#             out_3 = out
#             out = F.relu(self.layer3[2].bn1((self.layer3[2].conv1(out))))
#             out = self.layer3[2].conv2(out)
#             s30_x = self.get_similarity_matrix(out)
#             out = F.relu(self.layer3[2].bn2(out))
#             out = self.layer3[2].bn3(self.layer3[2].conv3(out))
#             out += self.layer3[2].shortcut(out_3)
#             out = F.relu(out)
            
#             out = self.layer3[3:5](out)
#             out_4 = out
#             out = self.layer3[5].conv1(out)
# #             #s38_x = self.get_similarity_matrix(out)
#             out = F.relu(self.layer3[5].bn1(out))
#             out = F.relu(self.layer3[5].bn2((self.layer3[5].conv2(out))))
#             out = self.layer3[5].conv3(out)
#             s40_x = self.get_similarity_matrix(out)
#             out = self.layer3[5].bn3(out)
#             out += self.layer3[5].shortcut(out_4)
#             out = F.relu(out)
#             out = F.avg_pool2d(out, 8)
#             out = out.view(out.size(0),-1)
                         
#             S = [ s10_x, s20_x, s30_x, s40_x]

#             if(img_type=="clean"):
#                 out = self.linear_clean(out)
#             elif(img_type=="noise"):
#                 out = self.linear_noise(out)
#             return (out, S)
        
        

### For Regularization Every Layer

            out, S = self.layer1((out, S))
            out, S = self.layer2((out, S))
            out, S = self.layer3((out, S))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            
            if(img_type=="clean"):
                out = self.linear_clean(out)
            elif(img_type=="noise"):
                out = self.linear_noise(out)
            return (out, torch.stack(S))
        else:
            out, _ = self.layer1((out, S))
            out, _ = self.layer2((out, S))
            out, _ = self.layer3((out, S))
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            if(img_type=="clean"):
                out = self.linear_clean(out)
            elif(img_type=="noise"):
                out = self.linear_noise(out)
            return out


class Bottleneck_Original(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_Original, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def get_similarity_matrix(self, x): #Compute cosine similarity matrix over image batch
        x = x.view(x.shape[0],-1)
        x_s = (x-x.mean(dim=1, keepdim=True))/x.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/x.shape[1]
        return sim
    
    def forward(self, x):
        out = self.conv1(x)
        #s1_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        #s2_x = self.get_similarity_matrix(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        #s3_x = self.get_similarity_matrix(out)
        out = self.bn3(out)
        out = F.relu(out)
#         if(len(self.shortcut)>0):
#             out2 = self.shortcut[0](x)
#             sSkip = self.get_similarity_matrix(out2)
#             out2 = self.shortcut[1](out2)
#             out+=out2
#             out = F.relu(out) 
#             return (out, S+[s1_x, s2_x, s3_x, sSkip])
#         else:
#             out += self.shortcut(x)
#             out = F.relu(out) 
#             return (out, S+[s1_x, s2_x, s3_x])  
        return out
class ResNet_Original(nn.Module):
    def __init__(self, depth, num_classes):
        super(ResNet_Original, self).__init__()
        self.in_planes = 16

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3,16)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    def get_similarity_matrix(self, x):
        y = x
        y = y.view(y.shape[0],-1)
        x_s = (y-y.mean(dim=1, keepdim=True))/y.std(dim=1,keepdim=True)
        sim = (x_s@x_s.t())/y.shape[1]
        return sim
    
    def forward(self, x, compute_similarity=False):
        S = []
        out = self.conv1(x)
        #S.append(self.get_similarity_matrix(out))
        out = F.relu(self.bn1(out))
        if compute_similarity:
#             out_s1 = out
#             out = F.relu(self.layer1[0].bn1(self.layer1[0].conv1(out)))
#             out = F.relu(self.layer1[0].bn2(self.layer1[0].conv2(out)))
#             out = self.layer1[0].bn3(self.layer1[0].conv3(out))
#             out_s1 = self.layer1[0].shortcut[0](out_s1)
#             sSkip1 = self.get_similarity_matrix(out_s1)
#             out_s1 = self.layer1[0].shortcut[1](out_s1)
#             out+=out_s1
#             out = F.relu(out)
                         
                         
            out = self.layer1[:2](out)
            out_1 = out
            out = F.relu(self.layer1[2].bn1(self.layer1[2].conv1(out)))
            out = F.relu(self.layer1[2].bn2(self.layer1[2].conv2(out)))
            out = self.layer1[2].conv3(out)
            s10_x = self.get_similarity_matrix(out)
            #print(s10_x)
            out = self.layer1[2].bn3(out)
            out+=self.layer1[2].shortcut(out_1)
            out = F.relu(out)
            
#             out_s2 = out
#             out = F.relu(self.layer2[0].bn1(self.layer2[0].conv1(out)))
#             out = F.relu(self.layer2[0].bn2(self.layer2[0].conv2(out)))
#             out = self.layer2[0].bn3(self.layer2[0].conv3(out))
#             out_s2 = self.layer2[0].shortcut[0](out_s2)
#             sSkip2 = self.get_similarity_matrix(out_s2)
#             out_s2 = self.layer2[0].shortcut[1](out_s2)
#             out+=out_s2
#             out = F.relu(out)           
                         
            
            out = self.layer2[:3](out)
            out_2 = out
            out = self.layer2[3].conv1(out)
            s20_x = self.get_similarity_matrix(out)
            out = F.relu(self.layer2[3].bn1(out))
            out = self.layer2[3].conv2(out)
            #s21_x = self.get_similarity_matrix(out)
            out = F.relu(self.layer2[3].bn2(out))
            out = self.layer2[3].bn3(self.layer2[3].conv3(out))
            out += self.layer2[3].shortcut(out_2)
            out = F.relu(out)

#             out_s3 = out
#             out = F.relu(self.layer3[0].bn1(self.layer3[0].conv1(out)))
#             out = F.relu(self.layer3[0].bn2(self.layer3[0].conv2(out)))
#             out = self.layer3[0].bn3(self.layer3[0].conv3(out))
#             out_s3 = self.layer3[0].shortcut[0](out_s3)
#             sSkip3 = self.get_similarity_matrix(out_s3)
#             out_s3 = self.layer3[0].shortcut[1](out_s3)
#             out+=out_s3
#             out = F.relu(out)
                         
            out = self.layer3[:2](out)
            out_3 = out
            out = F.relu(self.layer3[2].bn1((self.layer3[2].conv1(out))))
            out = self.layer3[2].conv2(out)
            s30_x = self.get_similarity_matrix(out)
            out = F.relu(self.layer3[2].bn2(out))
            out = self.layer3[2].bn3(self.layer3[2].conv3(out))
            out += self.layer3[2].shortcut(out_3)
            out = F.relu(out)
            
            out = self.layer3[3:5](out)
            out_4 = out
            out = self.layer3[5].conv1(out)
            #s38_x = self.get_similarity_matrix(out)
            out = F.relu(self.layer3[5].bn1(out))
            out = F.relu(self.layer3[5].bn2((self.layer3[5].conv2(out))))
            out = self.layer3[5].conv3(out)
            s40_x = self.get_similarity_matrix(out)
            out = self.layer3[5].bn3(out)
            out += self.layer3[5].shortcut(out_4)
            out = F.relu(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0),-1)
            #print(out.shape)
            out = self.linear(out)
            S = [s10_x, s20_x, s30_x, s40_x]
            return (out, S)

        else:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, 8)
            out = out.view(out.size(0), -1)
            out = self.linear(out)

            return out

        
        
        
if __name__ == '__main__':
    net=ResNet(50, 10)
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())
