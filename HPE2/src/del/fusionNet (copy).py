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
import argparse

sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.src.model.deepPrior import dpnet
from HIG.src.model.network import Pix2pix

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/HPE/src/')
import loss as losses

class ReconstructionLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ReconstructionLayer,self).__init__()        
        self.fc=nn.Linear(input_dim,output_dim)
        
    def initializeLayer(self,w,b):
        self.fc.weight=torch.nn.Parameter(w)
        self.fc.bias=torch.nn.Parameter(b)
   
    def forward(self,x):    
        out=self.fc(x)
        return out
    
'''
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
        
        #return torch.sigmoid(out)
        return out
'''

class SkeletonDiscriminator(nn.Module):
    def __init__(self,input_dim):
        super(SkeletonDiscriminator,self).__init__()
        
        self.layer1=nn.Sequential(
                    nn.Linear(input_dim,1024),
                    #nn.InstanceNorm1d(1024),
                    nn.LeakyReLU(0.2, True))
        
        self.layer2=nn.Sequential(
                    nn.Linear(1024,1024),
                    #nn.InstanceNorm1d(1024),
                    nn.LeakyReLU(0.2,True),
                    nn.Linear(1024,1024))
        
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


class FusionNet(nn.Module):
    def __init__(self,pix2pix,hpe1,hpe2,args):
        super(FusionNet, self).__init__()
        self.net_hig=pix2pix.netG.to(args.device)
        self.net_hpe1=hpe1.to(args.device)
        self.net_hpe2=hpe2.to(args.device)
        
        if args.discriminator_reconstruction==True:
            net_hpd=SkeletonDiscriminator(args.skeleton_orig_dim)
        else:
            net_hpd=SkeletonDiscriminator(args.skeleton_pca_dim)
        self.net_hpd=net_hpd.to(args.device)
        self.net_recon=ReconstructionLayer(args.skeleton_pca_dim,args.skeleton_orig_dim)
        
        
        
        self.device=args.device
        self.is_train=args.is_train
        self.args=args
        
        self.out={}
        self.out['hig']=[]
        self.out['hpe1']=[]
        self.out['hig_hpe1']=[]
        self.out['hpe2']=[]
        self.out['hpe1_hpd']=[]
        self.out['hig_hpe1_hpd']=[]
        self.out['hpe2_hpd']=[]
        
        #to print loss
        self.loss_D_real={}
        self.loss_D_fake={}
        self.loss_D_real['hpe2']=None
        self.loss_D_fake['hpe2']=None
        self.loss_D_real['hig_hpe1']=None
        self.loss_D_fake['hig_hpe1']=None
        
        
        if self.is_train:
            #self.criterionL1 = torch.nn.L1Loss()
            
            self.optimizer_hig=torch.optim.Adam(self.net_hig.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
        
            self.optimizer_hpe2=torch.optim.Adam(self.net_hpe2.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
        
            self.optimizer_hpd=torch.optim.Adam(self.net_hpd.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
        
        
            self.JointsMSELoss= losses.JointsMSELoss().to(self.device)
            self.criterionGAN = losses.GANLoss(args.gan_mode).to(self.device)
            #self.criterionGAN= nn.BCELoss
    
    def set_input_icvl(self,img,hpose):
        img_icvl=torch.FloatTensor(img)
        hpose_icvl=torch.FloatTensor(hpose)
        
        self.img_icvl=img_icvl.to(self.device)
        self.hpose_icvl=hpose_icvl.to(self.device)
     
    def set_input(self,ir,depth):
        ir=torch.FloatTensor(ir)
        depth=torch.FloatTensor(depth)
        self.ir=ir.to(self.device)
        self.depth=depth.to(self.device)
        
    def set_mode(self,opt):
        if opt=='train':
            self.net_hig.train()
            self.net_hpe1.eval() #self.hpe1.train()
            self.net_hpe2.train()
            self.net_hpd.train()
            self.net_recon.eval()
            
        elif opt=='eval':
            self.net_hig.eval()
            self.net_hpe1.eval()
            self.net_hpe2.eval()
            self.net_hpd.eval()
            self.net_recon.eval()
        
    def forward(self):
        out1=self.net_hig(self.ir)
        out1=self.net_hpe1(out1)
        
        out2=self.net_hpe2(self.ir)
        return out1,out2
    
    def set_reconstruction_net(self,w,b):
        w=torch.FloatTensor(w)
        b=torch.FloatTensor(b)
        self.net_recon.initializeLayer(w,b)
        
    def reconstruct_joints(self,pca,com3d,cube,netname,option):  
        if netname=='hpe1':
            output_embed=self.out['hpe1']
        elif netname=='hig_hpe1':
            output_embed=self.out['hig_hpe1']
        elif netname=='hpe2':
            output_embed=self.out['hpe2']
             
        pca_w=torch.FloatTensor(pca.components_)
        pca_w=pca_w.to(self.device)
        pca_b=torch.FloatTensor(pca.mean_)
        pca_b=pca_b.to(self.device)
        output_recon=torch.mm(output_embed,pca_w)+pca_b      
        
        if option=='tocpu':
            output_recon=output_recon.detach().cpu().numpy()              
            output_recon=output_recon*(cube[2]/2.)+np.tile(com3d,(1,21))
                
        return output_recon
     
        
    def forward_(self,option):
        if option=='hpe1':
            self.out['hpe1']=self.net_hpe1(self.depth)        
        elif option=='hig_hpe1':
            self.out['hig']=self.net_hig(self.ir)
            
            self.out['hig_hpe1']=self.net_hpe1(self.out['hig'])
            self.out['hig_hpe1_recon']=self.net_recon(self.out['hig_hpe1'])
            
        elif option=='hpe2':
            self.out['hpe2']=self.net_hpe2(self.ir)
            self.out['hpe2_recon']=self.net_recon(self.out['hpe2'])
            
        elif option=='full':
            self.out['hpe1']=self.net_hpe1(self.depth)     
            self.out['hig']=self.net_hig(self.ir)
            self.out['hig_hpe1']=self.net_hpe1(self.out['hig']) 
            
            #
            self.out['hpe2']=self.net_hpe2(self.ir)
            
    def calculateloss(self):
        reconstruct=self.args.discriminator_reconstruction
        
        #(1)--hig
        self.forward_('hpe1')
        self.forward_('hig_hpe1')
        self.backward_D('hig_hpe1','no',reconstruct)
        self.backward_HIG('no',reconstruct)
        
        #(2)--hpe2
        self.forward_('hpe1')
        self.forward_('hpe2')
        self.backward_D('hpe2','no',reconstruct)
        self.backward_HPE2('no',reconstruct)
        

    def optimize_parameters(self):
        reconstruct=self.args.discriminator_reconstruction
 
        #(1)--optimize hig
        self.forward_('hpe1')
        self.forward_('hig_hpe1')
        
        #update hpd
        if self.args.discriminator_skeleton==True:
            self.set_requires_grad(self.net_hpd,True)
            self.optimizer_hpd.zero_grad()
            self.backward_D('hig_hpe1','backward',reconstruct)
            self.optimizer_hpd.step()
        
        #update hig
        self.set_requires_grad(self.net_hpd,False)
        self.optimizer_hig.zero_grad()
        self.backward_HIG('backward',reconstruct)
        self.optimizer_hig.step()
        
        #(2)--optimizer hpe2
        self.forward_('hpe1')
        self.forward_('hpe2')
        
        #update hpd
        if self.args.discriminator_skeleton==True:
            self.set_requires_grad(self.net_hpd,True)
            self.optimizer_hpd.zero_grad()
            self.backward_D('hpe2','backward',reconstruct)
            self.optimizer_hpd.step()
        
        #update hpe2
        self.set_requires_grad(self.net_hpd,False)
        self.optimizer_hpe2.zero_grad()
        self.backward_HPE2('backward',reconstruct)
        self.optimizer_hpe2.step()
            
    def backward_D(self,outname,option,reconstruct):
        if reconstruct==True:
            outname=outname+'_recon'
            hpose_real=self.net_recon(self.hpose_icvl)
        else:
            hpose_real=self.hpose_icvl
        
        #fake
        pred_fake=self.net_hpd(self.out[outname].detach())  
        self.loss_D_fake[outname]=self.criterionGAN(pred_fake,False) 
        
        #real
        true_real=self.net_hpd(hpose_real)
        self.loss_D_real[outname]=self.criterionGAN(true_real,True)
        
        
        #combine loss of real and fake
        self.loss_D=(self.loss_D_real[outname]+self.loss_D_fake[outname])*0.5

        if option=='backward':
            self.loss_D.backward()
    
    
    def backward_HIG(self,option,reconstruct):
        if reconstruct==True:
            pred_fake=self.net_hpd(self.out['hig_hpe1_recon'])
        else:
            pred_fake=self.net_hpd(self.out['hig_hpe1'])
    
        self.loss_hig_hpe1_mse=self.JointsMSELoss(self.out['hig_hpe1'],self.out['hpe1'])
        
        #before
        '''
        if self.args.discriminator_skeleton==True:
            self.loss_hig_hpe1_gan=self.criterionGAN(pred_fake,True)
            self.loss_hig_hpe1= self.args.lambda_L1*self.loss_hig_hpe1_mse + self.loss_hig_hpe1_gan
        else:
            self.loss_hig_hpe1= self.loss_hig_hpe1_mse 
        '''
        #now
        self.loss_hig_hpe1= self.args.lambda_L1*self.loss_hig_hpe1_mse
        
        if self.args.discriminator_skeleton==True:
            self.loss_hig_hpe1_gan=self.criterionGAN(pred_fake,True)
            self.loss_hig_hpe1+=self.loss_hig_hpe1_gan
            
        if self.args.generator_depthimg==True:
            self.loss_hig_gan=self.
            self.loss_hig_hpe1+=self.loss_hig_gan
            
        
        if option=='backward':
            self.loss_hig_hpe1.backward()
        
    def backward_HPE2(self,option,reconstruct):
        if reconstruct==True:
            pred_fake=self.net_hpd(self.out['hpe2_recon'])
        else:
            pred_fake=self.net_hpd(self.out['hpe2'])
    
        self.loss_hpe2_mse=self.JointsMSELoss(self.out['hpe2'],self.out['hpe1'])
       
        #before
        '''
        if self.args.discriminator_skeleton==True:
            self.loss_hpe2_gan=self.criterionGAN(pred_fake,True)           
            self.loss_hpe2= self.args.lambda_L1*self.loss_hpe2_mse + self.loss_hpe2_gan
        else:
            self.loss_hpe2= self.loss_hpe2_mse
        '''
        #now
        self.loss_hpe2= self.args.lambda_L1*self.loss_hpe2_mse
        
        if self.args.discriminator_skeleton==True:
            self.loss_hpe2_gan=self.criterionGAN(pred_fake,True)           
            self.loss_hpe2+= self.loss_hpe2_gan
        
        if option=='backward':
            self.loss_hpe2.backward()
   
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad    
        
    def getloss(self,loss_names):
        out=[]
        if 'hig_hpe1_mse' in loss_names:
            out.append(self.loss_hig_hpe1_mse.item())
        if 'hig_hpe1_gan' in loss_names:
            out.append(self.loss_hig_hpe1_gan.item())
        if 'hig_hpe1_real' in loss_names:
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_D_real['hig_hpe1_recon'].item())
            else:
                out.append(self.loss_D_real['hig_hpe1'].item())
        if 'hig_hpe1_fake' in loss_names:
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_D_fake['hig_hpe1_recon'].item())
            else:
                out.append(self.loss_D_fake['hig_hpe1'].item())
        
        
        if 'hpe2_mse' in loss_names:
            out.append(self.loss_hpe2_mse.item())
        if 'hpe2_gan' in loss_names:
            out.append(self.loss_hpe2_gan.item())
        if 'hpe2_real' in loss_names:
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_D_real['hpe2_recon'].item())
            else:
                out.append(self.loss_D_real['hpe2'].item())
        if 'hpe2_fake' in loss_names:
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_D_fake['hpe2_recon'].item())
            else:
                out.append(self.loss_D_fake['hpe2'].item())
        
        return out
        
        
        '''
        o1=self.loss_hig_hpe1_mse.item()
        o2=self.loss_hig_hpe1_gan.item()
        
        if self.args.discriminator_reconstruction==True:
            o3=self.loss_D_real['hig_hpe1_recon'].item()
            o4=self.loss_D_fake['hig_hpe1_recon'].item()
        else:
            o3=self.loss_D_real['hig_hpe1'].item()
            o4=self.loss_D_fake['hig_hpe1'].item()

        o5=self.loss_hpe2_mse.item()
        o6=self.loss_hpe2_gan.item()        
        
        if self.args.discriminator_reconstruction==True:
            o7=self.loss_D_real['hpe2_recon'].item()
            o8=self.loss_D_fake['hpe2_recon'].item()
        else:
            o7=self.loss_D_real['hpe2'].item()
            o8=self.loss_D_fake['hpe2'].item()
        
        
        return o1,o2,o3,o4,o5,o6,o7,o8
        '''
        
    
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch hand pose Training')
    args=parser.parse_args()

    #--user setting
    
    
    #--common setting
    args.solver='adam'
    args.epochs=100
    args.train_batch=5
    args.lr=5.0e-6
    args.momentum=0
    args.weight_decay=1.0e-5
    args.gamma=0.9
    args.beta1=0.5
    args.is_train=True
    args.gan_mode='vanilla'
    args.gpu_ids='0'
    args.lambda_L1=50
    args.trainImageSize=128
    args.skeleton_pca_dim=52
    args.skeleton_orig_dim=63
    
    
    #device
    device = torch.device("cuda:%s"%args.gpu_ids if torch.cuda.is_available() else "cpu")
    args.device=device
    
    
    #network
    
    pix2pix=Pix2pix(args)
    
    hpe1=dpnet(52,5,device)
    hpe2=dpnet(52,5,device)
    net=FusionNet(pix2pix,hpe1,hpe2,args)
    
    w=np.random.rand(args.skeleton_orig_dim,args.skeleton_pca_dim)
    b=np.random.rand(args.skeleton_orig_dim)
    net.set_reconstruction_net(w,b)
    
    net=net.to(device)
   
    #run
    ir=np.random.rand(args.train_batch,1,128,128)    
    depth=np.random.rand(args.train_batch,1,128,128)    
    
    net.set_input(ir,depth)
    
    hpose_icvl=np.random.rand(args.train_batch,args.skeleton_pca_dim)
    img_icvl=np.random.rand(args.train_batch,1,128,128)
    net.set_input_icvl(img_icvl,hpose_icvl)
    
    
    
    #train
    net.optimize_parameters()
    
    
    
    '''
    net=ReconstructionLayer(52,63)
    w=np.random.rand(63,52)
    b=np.random.rand(63)
    w=torch.FloatTensor(w)
    b=torch.FloatTensor(b)
    
    net.initializeLayer(w,b)
    
    
    x=torch.FloatTensor(np.random.rand(3,52))
    out=net(x)
    print(out.shape)
    '''
    
    
    
    
