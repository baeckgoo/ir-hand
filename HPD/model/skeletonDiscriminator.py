import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class SkeletonDiscriminator(nn.Module):
    def __init__(self,input_dim):
        super(SkeletonDiscriminator,self).__init__()
        
        self.input_dim=input_dim      
    
        self.layer1=nn.Sequential(
                    nn.Conv2d(input_dim, 128, kernel_size=1, bias=True),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, True))
          
        self.layer2=nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=1, bias=True),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, True))
        
        self.layer3=nn.Sequential(
                    nn.Conv2d(256, 512, kernel_size=1, bias=True),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, True))
        
        
        self.layer4=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=1, bias=True),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, True))
        
        self.layer5=nn.Sequential(
                    nn.Conv2d(1024, 1, kernel_size=1, bias=True))
        
    
    def forward(self,input):
        input=input.view(-1,self.input_dim,1,1)
        out1=self.layer1(input)
        out2=self.layer2(out1)
        out3=self.layer3(out2)
        out4=self.layer4(out3)
        out=self.layer5(out4)
        
        return out
        #return torch.sigmoid(out)
        
        
'''
class SkeletonDiscriminator(nn.Module):
    def __init__(self,input_dim):
        super(SkeletonDiscriminator,self).__init__()
        
        self.layer1=nn.Sequential(
                    nn.Linear(input_dim,128),
                    #nn.InstanceNorm1d(128),
                    nn.LeakyReLU(0.2, True))
        
        self.layer2=nn.Sequential(
                    nn.Linear(128,256),
                    #nn.InstanceNorm1d(1024),
                    nn.LeakyReLU(0.2,True),
                    nn.Linear(256,512))
                    
                    nn.LeakyReLU(0.2,True),
                    nn.Linear(512,1024))
        
        self.layer3=nn.Sequential(
                    nn.LeakyReLU(0.2,True),
                    nn.Linear(1024,1))
    
    def forward(self,input):
        out1=self.layer1(input)
        out2=self.layer2(out1)
        out3=out1+out2
        out=self.layer3(out3)
        
        return out
        #return torch.sigmoid(out)
'''


