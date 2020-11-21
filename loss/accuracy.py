# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Wei Yang (platero.yang@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np


class Accuracy(nn.Module):
    def __init__(self,pca_w,pca_b,cube,jointNum,batch_size,device):
        super(Accuracy, self).__init__()
        #self.criterion = nn.MSELoss()
        #self.criterion = nn.MSELoss(reduction='elementwise_mean')
        
        self.pca_w=pca_w
        self.pca_b=pca_b
        self.cube=cube
        
        self.com3d=np.zeros((batch_size,3))
        self.joints3d_gt=np.zeros((batch_size,jointNum*3))
        
        self.batch_size=batch_size
        
        self.device=device
        
        

    def _forward(self, output_embed, target,com3d):
        
        output_recon=torch.mm(output_embed,self.pca_w)+self.pca_b
        
        com3d_tile=np.tile(com3d,(1,21))
        com3d_tile=torch.FloatTensor(com3d_tile)          
        com3d_tile=  com3d_tile.to(self.device)                 
        output_recon=output_recon*(self.cube[2]/2.)+com3d_tile
                
        error_bag=[]
        for k in range(self.batch_size):
            error_torch1=(target[k]-output_recon[k])**2
            error_torch2=torch.sqrt(torch.sum(error_torch1.view(21*1,3),dim=1))       
            error=torch.sum(error_torch2)/(21*1)
            error_bag.append(error.item())
                    
        #loss=self.criterion(output,target)
        #error=self.calculate_error()
        return error_bag
        #return error

        
        