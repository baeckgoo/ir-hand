import pyrealsense2 as rs
import os,sys,inspect
currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
basedir=currentdir+'\\..\\'
sys.path.append(basedir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import numpy as np
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import time
import argparse

#sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
from HPE.model.deepPrior import dpnet
from HIG.model.network import Pix2pix
from HIG.model.network import Discriminator as HandDepthImageDiscriminator
from HPD.model.skeletonDiscriminator import SkeletonDiscriminator

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

class FusionNet(nn.Module):
    def __init__(self,hpe1_orig,pix2pix,hpe1,hpe2,hpe1_refine,args):
        super(FusionNet, self).__init__()
        self.args=args
        

        if pix2pix!=None: self.net_hig=pix2pix.netG.to(args.device)
        if hpe1!=None:    self.net_hpe1=hpe1.to(args.device)
        if hpe2!=None:    self.net_hpe2=hpe2.to(args.device)
        if hpe1_orig!=None: self.net_hpe1_orig=hpe1_orig.to(args.device)
        if hpe1_refine!=None: self.net_hpe1_refine=hpe1_refine.to(args.device)
        
        
        if 'hpd' in args.use_net:
            if 'recon' in args.use_net: 
                net_hpd=SkeletonDiscriminator(args.skeleton_orig_dim)
            else:
                net_hpd=SkeletonDiscriminator(args.skeleton_pca_dim)           
            self.net_hpd=net_hpd.to(args.device)     
            
        if 'hid' in args.use_net:
            self.net_hid=pix2pix.netD.to(args.device)
        if 'recon' in args.use_net:        
            self.net_recon=ReconstructionLayer(args.skeleton_pca_dim,args.skeleton_orig_dim)
        

        self.device=args.device
        self.is_train=args.is_train
        
        
        self.out={}
        self.out['hpe1_orig']=[]
        self.out['hig']=[]
        self.out['hig_hpe1']=[]
        self.out['hpe2']=[]
        self.out['hpe1_refine']=[]
        
        self.out['hpe1_orig_hpd']=[]
        self.out['hig_hpe1_hpd']=[]
        self.out['hpe2_hpd']=[]
        self.out['hpe1_refine_hpd']=[]
        
        #to print loss
        self.loss_hpd_real={}
        self.loss_hpd_fake={}
        self.loss_hpd_real['hpe1_orig']=None
        self.loss_hpd_real['hig_hpe1']=None
        self.loss_hpd_real['hpe2']=None
        self.loss_hpd_real['hpe1_refine']=None
        self.loss_hpd_fake['hpe1_orig']=None
        self.loss_hpd_fake['hig_hpe1']=None
        self.loss_hpd_fake['hpe2']=None
        self.loss_hpd_fake['hpe1_refine']=None
        #
        self.use_blurdata=False
        self.selection_idx=0
                
        #
        if self.is_train:
            if 'hpe1' in args.train_net:
                self.optimizer_hpe1=torch.optim.Adam(self.net_hpe1.parameters(),
                                        lr=args.lr_hpe1,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
            if 'hpe1_refine' in args.train_net:
                self.optimizer_hpe1_refine=torch.optim.Adam(self.net_hpe1_refine.parameters(),
                                        lr=args.lr_hpe1_refine,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
            if 'hig' in args.train_net:
                self.optimizer_hig=torch.optim.Adam(self.net_hig.parameters(),
                                        lr=args.lr_hig,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
            if 'hpe2' in args.train_net:
                self.optimizer_hpe2=torch.optim.Adam(self.net_hpe2.parameters(),
                                        lr=args.lr_hpe2,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
            if 'hpd' in args.train_net:            
                self.optimizer_hpd=torch.optim.Adam(self.net_hpd.parameters(),
                                        lr=args.lr_hpd,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
            if 'hid' in args.train_net:
                self.optimizer_hid=torch.optim.Adam(self.net_hid.parameters(),
                                        lr=args.lr_hid,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
        
        
            self.JointsMSELoss= losses.JointsMSELoss().to(self.device)
            self.criterionGAN = losses.GANLoss(args.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            #self.criterionGAN= nn.BCELoss
            
    def set_parameters_hpd(self,param):
        self.net_hpd.load_state_dict(param)
        
        
    
    def set_input_icvl(self,img,hpose):
        img_icvl=torch.FloatTensor(img)
        hpose_icvl=torch.FloatTensor(hpose)
        
        self.depth=img_icvl.to(self.device)
        self.hpose_icvl=hpose_icvl.to(self.device)
     
    def set_input(self,ir,depth):               
        ir=torch.FloatTensor(ir)
        depth=torch.FloatTensor(depth)
        self.ir=ir.to(self.device)
        self.depth=depth.to(self.device)
        
        
    def set_mode(self,opt):
        if opt=='train':
            if 'hig' in self.args.use_net:
                if 'hig' in self.args.train_net:
                    self.net_hig.train()
                else:   
                    self.net_hig.eval()                
            if 'hpe1' in self.args.use_net:
                if 'hpe1' in self.args.train_net:
                    self.net_hpe1.train()
                else:
                    self.net_hpe1.eval()
            if 'hpe1_refine' in self.args.use_net:
                if 'hpe1_refine' in self.args.train_net:
                    self.net_hpe1_refine.train()
                else:
                    self.net_hpe1_refine.eval()
            if 'hpe2' in self.args.use_net:
                if 'hpe2' in self.args.train_net:
                    self.net_hpe2.train()
                else:
                    self.net_hpe2.eval()
            
            if 'hpd' in self.args.use_net:
                if 'hpd' in self.args.train_net:
                    self.net_hpd.train()
                else:
                    self.net_hpd.eval()
            
            if 'hid' in self.args.use_net:
                if 'hid' in self.args.train_net:
                    self.net_hid.train()
                else:
                    self.net_hid.eval()
            
            if 'recon' in self.args.use_net:
                self.net_recon.eval()
            if 'hpe1_orig' in self.args.use_net:
                self.net_hpe1_orig.eval()    

        elif opt=='eval':
            if 'hig' in self.args.use_net:
                self.net_hig.eval()
            if 'hpe1' in self.args.use_net:
                self.net_hpe1.eval()
            if 'hpe2' in self.args.use_net:
                self.net_hpe2.eval()
            if 'hpd' in self.args.use_net:
                self.net_hpd.eval()
            if 'hid' in self.args.use_net:
                self.net_hid.eval()
            if 'recon' in self.args.use_net:
                self.net_recon.eval()
            if 'hpe1_orig' in self.args.use_net:
                self.net_hpe1_orig.eval()   
            if 'hpe1_refine' in self.args.use_net:
                self.net_hpe1_refine.eval() 
                
    '''    
    def forward(self):
        out1=self.net_hig(self.ir)
        out1=self.net_hpe1(out1)
        
        out2=self.net_hpe2(self.ir)
        return out1,out2
    '''
    
    def set_reconstruction_net(self,w,b):
        w=torch.FloatTensor(w)
        b=torch.FloatTensor(b)
        self.net_recon.initializeLayer(w,b)
        
    def reconstruct_joints(self,pca,com3d,cube,netname,option):  
        if netname=='hpe1_orig':
            output_embed=self.out['hpe1_orig']
        elif netname=='hig_hpe1':
            output_embed=self.out['hig_hpe1']
        elif netname=='hpe2':
            output_embed=self.out['hpe2']
        elif netname=='hpe1_refine':
            output_embed=self.out['hpe1_refine']
            
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
        if option=='hpe1_orig':
            self.out['hpe1_orig']=self.net_hpe1_orig(self.depth)        
        elif option=='hpe1_orig_with_inter':
            fm,o=self.net_hpe1_orig.forward_with_inter(self.depth) 
            self.out['hpe1_orig']=o
            self.out['hpe1_orig_inter']=fm          
        elif option=='hig_hpe1':
            self.out['hig']=self.net_hig(self.ir)  
            self.out['hig_hpe1']=self.net_hpe1(self.out['hig'])
            #self.out['hig_hpe1_recon']=self.net_recon(self.out['hig_hpe1'])    
        elif option=='hig_hpe1_with_inter':
            self.out['hig']=self.net_hig(self.ir)
            fm,o=self.net_hpe1.forward_with_inter(self.out['hig'])
            self.out['hig_hpe1']=o
            self.out['hig_hpe1_inter']=fm      
        elif option=='hpe2':
            self.out['hpe2']=self.net_hpe2(self.ir)
        elif option=='hpe2_with_inter':
            fm,o=self.net_hpe2.forward_with_inter(self.ir)
            self.out['hpe2']=o
            self.out['hpe2_inter']=fm
        elif option=='hpe1_refine':
            self.out['hpe1_refine']=self.net_hpe1_refine(self.depth)
        elif option=='hpe1_refine_with_inter':
            fm,o=self.net_hpe1_refine.forward_with_inter(self.depth)
            self.out['hpe1_refine']=o
            self.out['hpe1_refine_inter']=fm
            
            #self.out['hpe2_recon']=self.net_recon(self.out['hpe2'])    
        '''    
        elif option=='full':
            self.out['hpe1']=self.net_hpe1(self.depth)     
            self.out['hig']=self.net_hig(self.ir)
            self.out['hig_hpe1']=self.net_hpe1(self.out['hig'])     
            self.out['hpe2']=self.net_hpe2(self.ir)
        '''
    
    def forward_hpd(self,name):
        if name=='hpe1':
            out=self.net_hpd(self.out['hpe1'])
        elif name=='hig_hpe1':
            out=self.net_hpd(self.out['hig_hpe1'])
        elif name=='hpe2':
            out=self.net_hpd(self.out['hpe2'])
        elif name=='hpe1_orig':
            out=self.net_hpd(self.out['hpe1_orig'])    
        elif name=='hpe1_refine':
            out=self.net_hpd(self.out['hpe1_refine']) 
            
        #return out.item()
        return out.item()
        
    def calculateloss_hig(self):
        self.forward_('hpe1_orig')
        self.forward_('hig_hpe1')
        
        if 'hid' in self.args.train_net:
            self.backward_HID('no')
        self.backward_HIG('no')
        
        
    def calculateloss_hpe2(self):  
        if self.use_blurdata==False:
            self.forward_('hpe1_orig')
        else:
            self.forward_('hig_hpe1')
        
        self.forward_('hpe2')
        self.backward_HPE2('no')
        
        #(3)--hpd
        '''
        if 'hpd' in self.args.use_net:
            if self.selection_idx%2==0:
                self.backward_HPD('hig_hpe1','no')
            else:
                self.backward_HPD('hpe2','no')
        
            #iter
            self.selection_idx+=1
            if self.selection_idx==2:
                self.selection_idx=0
        '''
    
    def calculateloss_hpe1_depthIR(self):   
        self.forward_('hpe1_orig')
        self.forward_('hpe1_refine')
        self.backward_HPE1_depthIR('no')
        
    def calculateloss_hpe1_bighand(self):        
        self.forward_('hpe1_refine')
        self.backward_HPE1_bighand('no')
        
        
    def optimize_parameters_hig(self):    
        #(1)--optimize hig with hpe1
        if self.args.train_with_inter==True:
            self.forward_('hpe1_orig_with_inter')
            self.forward_('hig_hpe1_with_inter')        
        else:
            self.forward_('hpe1_orig')
            self.forward_('hig_hpe1')
        
        #update hid
        if 'hid' in self.args.train_net:
            self.set_requires_grad(self.net_hid,True)
            self.optimizer_hid.zero_grad()
            self.backward_HID('backward')
            self.optimizer_hid.step()
        
        #update hig
        self.set_requires_grad(self.net_hid,False)
        self.optimizer_hig.zero_grad()
        self.backward_HIG('backward')
        self.optimizer_hig.step()

    def optimize_parameters_hpe2(self):
        if self.use_blurdata==False:
            if self.args.train_with_inter==True:
                self.forward_('hpe1_orig_with_inter')
                self.forward_('hpe2_with_inter')
            else:
                self.forward_('hpe1_orig') 
                self.forward_('hpe2')
        else:                 
            if self.args.train_with_inter==True:
                self.forward_('hig_hpe1_with_inter')
                self.forward_('hpe2_with_inter')                
            else:
                self.forward_('hig_hpe1')
                self.forward_('hpe2')
        
        self.optimizer_hpe2.zero_grad()             
        self.backward_HPE2('backward')
        self.optimizer_hpe2.step()
       
        
    def optimize_parameters_hpe1_depthIR(self):
        self.forward_('hpe1_orig')
        self.forward_('hpe1_refine')
        
        self.optimizer_hpe1_refine.zero_grad()
        self.backward_HPE1_depthIR('backward')
        self.optimizer_hpe1_refine.step()
    
    def optimize_parameters_hpe1_bighand(self):
        #self.img_icvl=img_icvl.to(self.device)
        #self.hpose_icvl=hpose_icvl.to(self.device)
        self.forward_('hpe1_refine')
        
        self.optimizer_hpe1_refine.zero_grad()
        self.backward_HPE1_bighand('backward')
        self.optimizer_hpe1_refine.step()
                
    def backward_HPD(self,fakename,option):
        #real (bighand)
        real=self.hpose_icvl
        pred_real=self.net_hpd(real)
        self.loss_hpd_real=self.criterionGAN(pred_real,True)
        
        #fake (the estimate)
        fake=self.out[fakename]
        pred_fake=self.net_hpd(fake)
        self.loss_hpd_fake=self.criterionGAN(pred_fake,False)
    
        #loss
        alpha=self.args.lambda_loss_hpd_real_icvl
        beta=self.args.lambda_loss_hpd_fake
        self.loss_hpd=alpha*self.loss_hpd_real.clone() + beta*self.loss_hpd_fake.clone()
        
        if option=='backward':
            self.loss_hpd.backward()

    def backward_HID(self,option):
        """Calculate GAN loss for the discriminator"""
        self.loss_hid=0
        
        # Fake; stop backprop to the generator by detaching fake_B
        #(1)gan
        if self.args.cgan==False:
            pred_fake=self.net_hid(self.out['hig'].detach())#gypark:IR->depth (ours)
        else:
            fake_AB=torch.cat((self.ir,self.out['hig']),1)
            pred_fake=self.net_hid(fake_AB.detach())
        
        self.loss_hid_fake = self.criterionGAN(pred_fake, False)
        self.loss_hid+=self.loss_hid_fake.clone()*self.args.lambda_loss_hid_fake
        
        # Real
        #(1)gan
        if self.args.cgan==False:
            pred_real = self.net_hid(self.depth)
        else:
            real_AB=torch.cat((self.ir,self.depth),1)
            pred_real = self.net_hid(real_AB)
        
        
        self.loss_hid_real = self.criterionGAN(pred_real, True)
        self.loss_hid+=self.loss_hid_real.clone() * self.args.lambda_loss_hid_real 
        
        # Real (ICVL)
        if self.args.lambda_loss_hid_real_icvl>0:
            pred_real_C = self.net_hid(self.img_icvl)
            self.loss_hid_real_icvl = self.criterionGAN(pred_real_C, True)
            # combine loss and calculate gradients
            self.loss_hid += self.loss_hid_real_icvl.clone()*self.args.lambda_loss_hid_real_icvl
                        
        if option=='backward':
            self.loss_hid.backward()
    '''    
    def backward_HPD(self,outname,option,reconstruct):
        self.loss_hpd=0
        
        if reconstruct==True:
            outname=outname+'_recon'
            hpose_real=self.net_recon(self.hpose_icvl)
        else:
            hpose_real=self.hpose_icvl
        
        #fake
        pred_fake=self.net_hpd(self.out[outname].detach())  
        self.loss_hpd_fake[outname]=self.criterionGAN(pred_fake,False) 
        self.loss_hpd+=self.loss_hpd_fake[outname]*self.args.lambda_loss_hpd_fake
        
        #real
        true_real=self.net_hpd(hpose_real)
        self.loss_hpd_real[outname]=self.criterionGAN(true_real,True)
        self.loss_hpd+=self.loss_hpd_real[outname]*self.args.lambda_loss_hpd_real_icvl
        

        if option=='backward':
            self.loss_hpd.backward()
    '''
    
    def backward_HIG(self,option):
        #hig-hpe1 mse
        self.loss_hig_hpe1_mse=self.JointsMSELoss(self.out['hig_hpe1'],self.out['hpe1_orig'])
        self.loss_hig=self.args.lambda_loss_hig_hpe1_mse*self.loss_hig_hpe1_mse.clone()

        #hig l1
        self.loss_hig_L1=self.criterionL1(self.out['hig'],self.depth)
        self.loss_hig+=self.loss_hig_L1.clone()*self.args.lambda_loss_hig_L1
        
        #loss with inter (intermediate)
        if self.args.train_with_inter==True:
            self.loss_hig_hpe1_inter=self.criterionL1(self.out['hig_hpe1_inter'],self.out['hpe1_orig_inter'])
            self.loss_hig+=self.loss_hig_hpe1_inter.clone()*self.args.lambda_loss_hig_hpe1_inter
        
        #depth image discriminator
        if 'hid' in self.args.train_net:
            #(1)gan
            if self.args.cgan==False:
                pred_fake=self.net_hid(self.out['hig'])
            else:
                fake_AB=torch.cat((self.ir,self.out['hig']),1)
                pred_fake=self.net_hid(fake_AB)
            
            self.loss_hig_gan = self.criterionGAN(pred_fake, True)   
            self.loss_hig+=self.loss_hig_gan.clone()*self.args.lambda_loss_hig_gan
        
        #hand skeleton discriminator 
        if 'hpd' in self.args.use_net:
            if 'recon' in self.args.use_net: 
                pred_fake=self.net_hpd(self.out['hig_hpe1_recon'])
            else:
                pred_fake=self.net_hpd(self.out['hig_hpe1'])    
            self.loss_hig_hpe1_gan=self.criterionGAN(pred_fake, True)   
            self.loss_hig+=self.loss_hig_hpe1_gan.clone()*self.args.lambda_loss_hig_hpe1_gan
            
        if option=='backward':
            self.loss_hig.backward()
                        
    def backward_HPE2(self,option):
        if 'hpd' in self.args.use_net:
            if 'recon' in self.args.use_net: 
                pred_fake=self.net_hpd(self.out['hpe2_recon'])
            else:
                pred_fake=self.net_hpd(self.out['hpe2'])
    
        if self.use_blurdata==True:
            self.loss_hpe2_mse=self.JointsMSELoss(self.out['hpe2'],self.out['hig_hpe1'])
        else:
            self.loss_hpe2_mse=self.JointsMSELoss(self.out['hpe2'],self.out['hpe1_orig'])
        self.loss_hpe2= self.args.lambda_loss_hpe2_mse*self.loss_hpe2_mse.clone()
        
        #loss with inter (intermediate)
        if self.args.train_with_inter==True:
            if self.use_blurdata==True:
                self.loss_hpe2_inter=self.criterionL1(self.out['hpe2_inter'],self.out['hig_hpe1_inter'])
                self.loss_hpe2+=self.loss_hpe2_inter.clone()*self.args.lambda_loss_hpe2_inter
            else:
                self.loss_hpe2_inter=self.criterionL1(self.out['hpe2_inter'],self.out['hpe1_orig_inter'])
                self.loss_hpe2+=self.loss_hpe2_inter.clone()*self.args.lambda_loss_hpe2_inter
            
    
        if 'hpd' in self.args.use_net:
            self.loss_hpe2_gan=self.criterionGAN(pred_fake,True)           
            self.loss_hpe2+= self.loss_hpe2_gan.clone()*self.args.lambda_loss_hpe2_gan
        
        if option=='backward':
            self.loss_hpe2.backward()
     
    
    def backward_HPE1_depthIR(self,option):
        self.loss_hpe1_refine_mse=self.JointsMSELoss(self.out['hpe1_orig'],self.out['hpe1_refine'])
        self.loss_hpe1_refine= self.loss_hpe1_refine_mse.clone()
        
        if option=='backward':
            self.loss_hpe1_refine.backward()
            
    def backward_HPE1_bighand(self,option):
        self.loss_hpe1_refine_mse=self.JointsMSELoss(self.hpose_icvl,self.out['hpe1_refine'])
        self.loss_hpe1_refine= self.loss_hpe1_refine_mse.clone()
        
        if option=='backward':
            self.loss_hpe1_refine.backward()
        
        
   
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
        #skselton (hig-hpe1)
        if 'hig_hpe1_mse' in loss_names:
            val=self.args.lambda_loss_hig_hpe1_mse*self.loss_hig_hpe1_mse.item()
            out.append(val) #ok
        if 'hig_hpe1_gan' in loss_names:
            val=self.args.lambda_loss_hig_hpe1_gan*self.loss_hig_hpe1_gan.item()
            out.append(val) #ok
          
        #skeleton (hpe2)  
        if 'hpe2_mse' in loss_names:
            val=self.args.lambda_loss_hpe2_mse*self.loss_hpe2_mse.item()
            out.append(val) #ok
        if 'hpe2_gan' in loss_names:
            val=self.args.lambda_loss_hpe2_gan*self.loss_hpe2_gan.item()
            out.append(val) #ok
            
        if 'hpe2_inter' in loss_names:
            val=self.args.lambda_loss_hpe2_inter*self.loss_hpe2_inter.item()
            out.append(val) #ok
            
        #depth (hig)
        if 'hig_L1' in loss_names:
            val=self.args.lambda_loss_hig_L1*self.loss_hig_L1.item()
            out.append(val) #ok
        if 'hig_gan' in loss_names:
            val=self.args.lambda_loss_hig_gan*self.loss_hig_gan.item()
            out.append(val) #ok
           
        #skeleton (hpe1_blur)
        if 'hpe1_refine_mse' in loss_names:
            val=self.args.lambda_loss_hpe1_refine_mse*self.loss_hpe1_refine_mse.item()
            out.append(val) #ok
        if 'hpe1_refine_gan' in loss_names:
            val=self.args.lambda_loss_hpe1_refine_gan*self.loss_hpe1_refine_gan.item()
            out.append(val) #ok
        if 'hpe1_refine_inter' in loss_names:
            val=self.args.lambda_loss_hpe1_refine_inter*self.loss_hpe1_refine_inter.item()
            out.append(val) #ok
        
        
        #feature map (hig-hpe1)
        if 'hig_hpe1_inter' in loss_names:
            val=self.args.lambda_loss_hig_hpe1_inter*self.loss_hig_hpe1_inter.item()
            out.append(val) #ok
   
        #depth discriminator (hid)
        if 'hid_real' in loss_names:
            val=self.args.lambda_loss_hid_real *self.loss_hid_real.item()
            out.append(val) #ok
        if 'hid_fake' in loss_names:
            val=self.args.lambda_loss_hid_fake*self.loss_hid_fake.item()
            out.append(val) #ok
        if 'hid_real_icvl' in loss_names:
            val=self.args.lambda_loss_hid_real_icvl*self.loss_hid_real_icvl.item()
            out.append(val) #ok


        #pose discriminator (hig-hpe1)
        if 'hpd_real' in loss_names:
            val=self.args.lambda_loss_hpd_real_icvl*self.loss_hpd_real.item()
            out.append(val)
            '''
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_hpd_real['hig_hpe1_recon'].item())
            else:
                out.append(self.loss_hpd_real['hig_hpe1'].item())
            '''
        if 'hpd_fake' in loss_names:
            val=self.args.lambda_loss_hpd_fake*self.loss_hpd_fake.item()
            out.append(val)
            '''
            if self.args.discriminator_reconstruction==True:
                out.append(self.loss_hpd_fake['hig_hpe1_recon'].item())
            else:
                out.append(self.loss_hpd_fake['hig_hpe1'].item())
            '''
        
        return out
        
    

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
    
    args.cgan=True
    
    
    
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
    
    
    
    
