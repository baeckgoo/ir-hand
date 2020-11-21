''' cite:   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import torch.backends.cudnn as cudnn
    
#from model.loss import GANLoss
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('/home/yong/dropbox/Dropbox-Uploader-master/gypark/')
import loss as losses

from loss.ganloss import GANLoss


class Identity(nn.Module):
    def forward(self, x):
        return x
        
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
   
###generator###
class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        #conv_one=nn.Conv2d(outer_nc,outer_nc,kernel_size=1,stride=1,bias=True)
        
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            #up = [uprelu, upconv, nn.Tanh()]
            #up = [uprelu, upconv]
            up = [uprelu, upconv, nn.Sigmoid()]
            
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


###discriminator###
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    

class Generator():
    def __init__(self,args, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        self.net=UnetGenerator(input_nc,output_nc,num_downs,ngf,norm_layer=norm_layer,use_dropout=False)
        
        
        
    
class Discriminator():
    def __init__(self,input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        self.net=NLayerDiscriminator(input_nc,ndf,n_layers,norm_layer=norm_layer)
      
        

class Pix2pix():
    def __init__(self,args,input_nc,output_nc):
        self.args=args
        
        generator=Generator(args,input_nc,output_nc,7,64,use_dropout=False)
        if args.cgan==True:
            discriminator=Discriminator(input_nc+output_nc,64,n_layers=3)
        else:
            discriminator=Discriminator(output_nc,64,n_layers=3)

        self.device = torch.device("cuda:0")
        cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

        self.is_train=args.is_train
        
        netG=generator.net
        netD=discriminator.net
        

        if args.segnet_weightfile!=None:
            print('initializing seg net..')
            checkpoint=torch.load(args.segnet_weightfile)          
            netG.load_state_dict(checkpoint['state_dict_pix2pix_G']) 
            netD.load_state_dict(checkpoint['state_dict_pix2pix_D']) 

            
        #G&D
        self.netG=netG.to(self.device)
        self.netD=netD.to(self.device)

        self.criterionGAN = GANLoss(args.gan_mode).to(self.device)
        #self.criterionL1 = torch.nn.L1Loss()
        #self.criterionL1 = torch.nn.CrossEntropyLoss()
        self.criterionL1= torch.nn.BCELoss()
    
    
        if self.is_train:
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=args.lr_generator,
                                        weight_decay=args.weight_decay,betas=(0.9,0.999))
        
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=args.lr_discriminator,
                                        weight_decay=args.weight_decay,betas=(0.9,0.999))
            
            self.scheduler_G = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G, milestones=[10,20,40,70], gamma=args.lr_decay_gamma)
            self.scheduler_D = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_D, milestones=[10,20,40,70], gamma=args.lr_decay_gamma)
       
        #losses
        self.losses={}
        
    def adjust_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
 
    def set_mode(self,opt):
        if opt=='train':
            self.netG.train()
            self.netD.train()
        else:
            self.netG.eval()
            self.netD.eval()
                
    def forward_test_phrase(self,ir_train,trainImageSize):
        img_hig = np.zeros((1,1,trainImageSize,trainImageSize), dtype = np.float32)
        img_hig[0,0,:,:]=np.copy(ir_train)
        self.set_input_test(img_hig)
        self.forward(self.real_A)
        _depth_pred=self.fake_B.detach().cpu().numpy()
        depth_pred=_depth_pred[0,0,:,:]
        return depth_pred
        
    
    '''    
    def set_input(self,img_s,img_t):
        img_s=torch.FloatTensor(img_s)
        img_t=torch.FloatTensor(img_t)
        #img_icvl=torch.FloatTensor(img_icvl)
        
        self.real_A = img_s.to(self.device)
        self.real_B = img_t.to(self.device)
        #self.real_C = img_icvl.to(self.device)
    '''   
    def set_input(self,x):
        img_s=torch.FloatTensor(x)
        self.real_A=img_s.to(self.args.device)
        
    def set_gt(self,y):
        img_t=torch.FloatTensor(y)
        self.real_B=img_t.to(self.args.device)
        
    
    def set_input_train(self,img_s,img_t):
        self.set_input(img_s)
        self.set_gt(img_t)
        
    def set_input_test(self,img_s):
        self.set_input(img_s)
            
    
    def forward(self,x):
        self.fake_B = self.netG(x)  # G(A)
        return self.fake_B
        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #(1)gan
        if self.args.cgan==False:
            pred_fake=self.netD(self.fake_B.detach())#gypark:IR->depth (ours)
        else:
            fake_AB=torch.cat((self.real_A, self.fake_B), 1) 
            pred_fake=self.netD(fake_AB.detach())#gypark:IR->depth (ours)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        # Real
        #(1) gan
        if self.args.cgan==False:
            pred_real = self.netD(self.real_B)
        else:
            real_AB=torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)        
        
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake*self.args.loss_weight['D_fake'] + self.loss_D_real * self.args.loss_weight['D_real'] 
        self.loss_D.backward()
    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #(1) gan
        if self.args.cgan==False:
            pred_fake=self.netD(self.fake_B) #gypark:cGAN->GAN
        else:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            pred_fake = self.netD(fake_AB)
        
        
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        #self.real_B=self.real_B.long()
        #print('max/min..',self.fake_B.max().item(),self.real_B.max().item(),self.fake_B.min().item(),self.real_B.min().item())
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B.squeeze(1)) 
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) 

        
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.args.loss_weight['G_L1']
        self.loss_G.backward()
    
    
    def optimize_parameters_G(self):
        #self.forward()
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

    def optimize_parameters_D(self):
        #self.forward()
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

    def train(self):
        self.forward(self.real_A)                   # compute fake images: G(A)
    
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        
        self.storeLoss()
        
    def evaluate(self):
        self.forward(self.real_A)                   # compute fake images: G(A)
        self.calculate_loss()    
        self.storeLoss()
        
        
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
                    
    def calculate_loss(self):
        #D
        #gan
        if self.args.cgan==False:
            pred_fake=self.netD(self.fake_B.detach())
        else:
        #cgan
            fake_AB=torch.cat((self.real_A, self.fake_B), 1) 
            pred_fake=self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #gan
        if self.args.cgan==False:
            pred_real = self.netD(self.real_B)
        else:
        #cgan
            real_AB=torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        
        
        #G
        #gan
        if self.args.cgan==False:
            pred_fake=self.netD(self.fake_B) #gypark:cGAN->GAN
        else:
        #cgan
            fake_AB=torch.cat((self.real_A, self.fake_B), 1) 
            pred_fake=self.netD(fake_AB) #gypark:cGAN->GAN
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        
        
    def storeLoss(self):
        '''store losses'''
        self.losses['G_L1']=self.args.loss_weight['G_L1'] * self.loss_G_L1#.item()
        self.losses['G_GAN']=self.loss_G_GAN#.item()
        self.losses['D_real']=self.args.loss_weight['D_real'] * self.loss_D_real#.item()
        self.losses['D_fake']=self.args.loss_weight['D_fake'] * self.loss_D_fake#.item() 

      
    def getloss(self):
        return self.losses
    
def getdataBatch(args,it,frame_idx,dataNum_list,filepath_list):
    '''batch'''
    data_batch=np.zeros((args.batch_size,1,args.cnnimg_size,args.cnnimg_size))
    label_batch=np.zeros((args.batch_size,3,args.cnnimg_size,args.cnnimg_size))
    
    for j in range(args.batch_size):
        frame=frame_idx[it*args.batch_size+j]        
        if frame>=0 and frame<dataNum_list[0]:
            datasetid=0
            frame_local=frame
        elif frame>=dataNum_list[0] and frame<sum(dataNum_list[:2]):
            datasetid=1
            frame_local=frame-dataNum_list[0]
        else:
            datasetid=2
            frame_local=frame-sum(dataNum_list[:2])
            
        depth=cv2.imread(filepath_list[datasetid]+"depth\\image_%05u_depth.png"%frame_local,2)
        depth=(depth-depth.min())/(depth.max()-depth.min())
        #depth=depth/1000
        
        label=cv2.imread(filepath_list[datasetid]+"label_filtered\\image_%05u_label.bmp"%frame_local,2)
        label2=cv2.resize(label,(args.cnnimg_size,args.cnnimg_size),interpolation=cv2.INTER_AREA)
                
        data_batch[j,0]=cv2.resize(depth,(args.cnnimg_size,args.cnnimg_size))
        
        label_batch[j,0][label2==255]=1
        label_batch[j,1][label2==0]=1
        label_batch[j,2][label2==1]=1
        
        '''
        label_batch[j,0][label2==255]=0
        label_batch[j,0][label2==0]=1
        label_batch[j,0][label2==1]=2
        '''
            
    return data_batch,label_batch
    

if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt
    import argparse
    import plotloss
    import tqdm
    
    parser = argparse.ArgumentParser()
    args= parser.parse_args()
    
    '''user setting'''
    debug=False
    args.batch_size=4#
    
    args.segnet_weightfile=None
    augmention_num=10
    
    save_filepath='..\\..\\weights\\segmentation_20201027_lr\\'
        
    '''common setting'''
    #filepath
    filepath_list=["D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user01\\",
                   "D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user02\\",
                   "D:\\research\\handdataset\\TwoPaintedHands\\paintedHands\\ego\\output_user04\\"]
    
    
    #parameters
    args.max_epoch=100
    args.is_train=True
    args.cgan=True
    args.gan_mode='vanilla'
    args.lr_generator=5e-4#1e-3
    args.lr_discriminator=1e-4#1e-3
    args.weight_decay=0#0.01
    args.lr_decay_step=20
    args.lr_decay_gamma=0.9
    args.device=0
    args.cnnimg_size=256

    
    save_epoch_iter=10
   
    dataNum_test=300
    dataNum_list=[3282,3109,3058-dataNum_test]
    
    dataNum_train=sum(dataNum_list)
    
    
    args.loss_weight={}
    args.loss_weight['G_L1']=100
    args.loss_weight['D_real']=0.5
    args.loss_weight['D_fake']=0.5
    
    
    #network
    pix2pix=Pix2pix(args,1,3)
    
    printed_loss_names=['G_L1','G_GAN','D_real','D_fake']
    plotloss_train=plotloss.Plotloss(printed_loss_names)
    plotloss_test=plotloss.Plotloss(printed_loss_names)
    
    #random
    rng = np.random.RandomState(23455)
    
    '''save arguments'''
    f=open(save_filepath+'\\setting.txt','w')
    for arg in vars(args):
        f.write('%s:%s\n'%(arg,getattr(args,arg)))
        print('%s:%s\n'%(arg,getattr(args,arg)))
    f.close()
    
    '''run'''
    data_batch=np.zeros((args.batch_size,1,args.cnnimg_size,args.cnnimg_size))
    
    frame_idx_train=np.arange(dataNum_train)
    frame_idx_test=np.arange(dataNum_train,dataNum_train+dataNum_test)
    np.random.shuffle(frame_idx_train)
    
    
    for epoch in range(args.max_epoch):
        '''train'''
        #np.random.shuffle(frame_idx_train)
        loop1 = tqdm.tqdm(range(dataNum_train//args.batch_size))
        for it in loop1:
            '''original image'''
            data_batch_orig,label_batch_orig=getdataBatch(args,it,frame_idx_train,dataNum_list,filepath_list)
            #plt.imshow(label_batch[0,0])
                    
            for aug_iter in range(augmention_num):
                #%%
                if aug_iter==0:
                    data_batch,label_batch=data_batch_orig,label_batch_orig
                
                if aug_iter>0:    
                    '''data augmentation'''
                    rot_range=30.
                    rot=rng.uniform(-rot_range,rot_range)
                    rot=np.mod(rot,360)
            
                    M=cv2.getRotationMatrix2D((0,0),-rot,1)
            
                    for bid in range(args.batch_size):
                        #plt.imshow(data_batch_orig[bid,0])
                        #plt.show()
                        img_seg_rot = cv2.warpAffine(data_batch_orig[bid,0], M, (args.cnnimg_size,args.cnnimg_size), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
                        data_batch[bid,0]=img_seg_rot
                        #plt.imshow(img_seg_rot)
                        #plt.show()
            
                        for j in range(3):
                            if j==0:
                                label_seg_rot = cv2.warpAffine(label_batch_orig[bid,j], M, (args.cnnimg_size,args.cnnimg_size), flags=cv2.INTER_LINEAR,
                                                            borderMode=cv2.BORDER_CONSTANT, borderValue=1)
                            else:
                                label_seg_rot = cv2.warpAffine(label_batch_orig[bid,j], M, (args.cnnimg_size,args.cnnimg_size), flags=cv2.INTER_LINEAR,
                                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                
                            label_batch[bid,j]=label_seg_rot        
                            #plt.imshow(label_seg_rot)
                            #plt.show()
                        #jio
                #%%
                '''training'''
                pix2pix.set_input_train(data_batch,label_batch)
                pix2pix.train()
            
            losses=pix2pix.getloss()
            message='(TRAIN)epc:%d:'%epoch
            for nm in printed_loss_names:
                plotloss_train.append_local(losses[nm].item(),nm)
                avg=plotloss_train.get_average(nm)   
                message+='%s(%.7f),'%(nm,avg)
            loop1.set_description(message)
            
            if debug:
                label_pred=pix2pix.fake_B.detach().cpu().numpy()[0,1]
                label_gt=pix2pix.real_B.detach().cpu().numpy()[0]
                plt.imshow(label_pred)
                plt.savefig('..\\..\\weights\\debug\\%d-%d.png'%(epoch,it))
    
        plotloss_train.append_loss()
        
        
        '''test'''
        loop2=tqdm.tqdm(range(dataNum_test//args.batch_size))
        for it2 in loop2:
            data_batch,label_batch=getdataBatch(args,it2,frame_idx_test,dataNum_list,filepath_list)
            
            #evaluate
            pix2pix.set_input_train(data_batch,label_batch)
            pix2pix.evaluate()
            
            losses=pix2pix.getloss()
            message='(TEST)epc:%d:'%epoch
            for nm in printed_loss_names:
                plotloss_test.append_local(losses[nm].item(),nm)
                avg=plotloss_test.get_average(nm)   
                message+='%s(%.7f),'%(nm,avg)
            loop2.set_description(message)     
        plotloss_test.append_loss()
        
        
        '''plot loss'''
        plotloss_train.save_plot('loss',save_filepath+'train_loss.png')
        plotloss_test.save_plot('loss',save_filepath+'test_loss.png')
        
        '''save weights'''
        if epoch>=0 and epoch%save_epoch_iter==0:
            state_best={
                        'epoch':epoch,
                        'state_dict_pix2pix_G':pix2pix.netG.state_dict(),
                        'optimizer_pix2pix_G':pix2pix.optimizer_G.state_dict(),
                        'state_dict_pix2pix_D':pix2pix.netD.state_dict(),
                        'optimizer_pix2pix_D':pix2pix.optimizer_D.state_dict()}                        
            torch.save(state_best,save_filepath+'segmentation-%d.pth.tar'%epoch)
            
            
            
        '''adjust lr'''
        if args.lr_decay_gamma>0:
            pix2pix.adjust_lr()
            
                
                
        
        
        #data_batch
        
    
    
    
    
    