'''
ref: https://github.com/bearpaw/pytorch-pose/blob/master/pose/models/hourglass_gn.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import os
import sys

import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import time

class Bottleneck(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        planes_reduced = planes // 4

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes_reduced, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes_reduced)
        self.conv2 = nn.Conv2d(planes_reduced, planes_reduced, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes_reduced)
        self.conv3 = nn.Conv2d(planes_reduced, planes , kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class DeepPriorNet(nn.Module):
    '''Deep prior net by obweger'''
    def __init__(self, block, num_classes=45,num_blocks=5,device=None):
        super(DeepPriorNet, self).__init__()
        print('num_classes is..',num_classes)
        self.device=device
        num_feats = [32,64,128,256,256]
        num_blocks=num_blocks  #test중(.. num_block현재 3임).. 5로 바꿔야됨.
        self.expected_fc_featurenum=8*8*256
        self.dropout_ratio=0.3
    
        conv1 = nn.Conv2d(1, num_feats[0], kernel_size=5, stride=1, padding=2,
                               bias=True)
        maxpool1 = nn.MaxPool2d(2, stride=2)


        layer1 = self._make_residual(block,num_feats[0],num_feats[1],num_blocks,stride=2)   
        layer2 = self._make_residual(block,num_feats[1],num_feats[2],num_blocks,stride=2)   
        layer3 = self._make_residual(block,num_feats[2],num_feats[3],num_blocks,stride=2)   
        layer4 = self._make_residual(block,num_feats[3],num_feats[4],num_blocks)   
        
        
        self.conv_module=nn.Sequential(
                        conv1,
                        maxpool1,
                        layer1,
                        layer2,
                        layer3,
                        layer4)
        
        
        
        self.relu = nn.ReLU(inplace=True)
        fc1=nn.Linear(self.expected_fc_featurenum,1024)
        self.fc_module1=nn.Sequential(
                        fc1,
                        self.relu)
        
        
        fc2=nn.Linear(1024,1024)
        self.fc_module2=nn.Sequential(
                        fc2,
                        self.relu)
        
        fc3=nn.Linear(1024,num_classes)
        self.fc_module3=nn.Sequential(
                        fc3)
        
    def _make_residual(self, block, inplanes,planes, blocks, stride=1):
        downsample = None
        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        
        for i in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)      
                

    def forward(self, x):
        out=self.conv_module(x)
        
        out=out.view(-1,self.expected_fc_featurenum)
        
        out=self.fc_module1(out)
        out=F.dropout(out,self.dropout_ratio,training=self.training)

        out=self.fc_module2(out)
        out=F.dropout(out,self.dropout_ratio,training=self.training)
        
        out=self.fc_module3(out)
        
        return out
    
    def forward_with_inter(self,x):
        out=self.conv_module(x)
        out_feature=out.clone()
        
        out=out.view(-1,self.expected_fc_featurenum)
        
        out=self.fc_module1(out)
        out=F.dropout(out,self.dropout_ratio,training=self.training)

        out=self.fc_module2(out)
        out=F.dropout(out,self.dropout_ratio,training=self.training)
        
        out=self.fc_module3(out)
        
        return out_feature,out
        
    

    def forward_with_reconstruction(self,x,pca,com3d,cube):
        #predict
        input=torch.FloatTensor(x)
        input=input.to(self.device)
        output_embed=self.forward(input)
                
        #reconstruct
        pca_w=torch.FloatTensor(pca.components_)
        pca_w=pca_w.to(self.device)
        pca_b=torch.FloatTensor(pca.mean_)
        pca_b=pca_b.to(self.device)
        output_recon=torch.mm(output_embed,pca_w)+pca_b            
        output_recon=output_recon.detach().cpu().numpy()
              
        output_recon=output_recon*(cube[2]/2.)+np.tile(com3d,(1,21))
                
        return output_recon
    
        
        

def dpnet(num_classes,num_blocks,device): 
    model = DeepPriorNet(Bottleneck,num_classes=num_classes,num_blocks=num_blocks,device=device)
                         
    return model

if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #model

    model = dpnet(52,5,device)
    model=model.to(device)
    #model = torch.nn.DataParallel(model).to(device)

    #optimizer
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=0.01,
                                    momentum=0,
                                    weight_decay=0.00)
        
    #data
    batchsize=1
    data=np.random.rand(batchsize,1,128,128)    
    gt_train=np.random.rand(batchsize,52)
        
    #model.train()
    model.eval()
    
    for i in range(1):
        input=torch.FloatTensor(data)
        input=input.to(device)
        
        target=torch.FloatTensor(gt_train)
        target=target.to(device)

        TIME=time.time()
        output=model(input)
        print('forward time:',time.time()-TIME)
        print('output shape:',output.shape)
        print('')
        #optimizer.zero_grad()
        
        #output=model(input)
        
        #loss=criterion(output,target)
        #error=evaluateEmbeddingError(output,target)
        
        #loss.backward()
        #optimizer.step()
        
        
        #print('loss:',loss.item())
        #print('error:',error.item())
        
        
    
    
    
    
    
    
    
    
