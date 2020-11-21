''' 
cite:   https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
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

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
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
    def __init__(self,args):
        self.args=args
        
        generator=Generator(args,1,1,7,64,use_dropout=False)
        if args.cgan==True:
            discriminator=Discriminator(2,64,n_layers=3)
        else:
            discriminator=Discriminator(1,64,n_layers=3)

        self.device = torch.device("cuda:"+args.gpu_ids if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33

        self.is_train=args.is_train
        
        #G&D
        if 'hig' in args.use_net:
            self.netG=generator.net.to(self.device)

        if 'hid' in args.use_net:
            self.netD=discriminator.net.to(self.device)

        if self.is_train:
            self.criterionGAN = losses.GANLoss(args.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
    
            if 'hig' in args.train_net:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                        lr=args.lr_hig,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
        
            if 'hid' in args.train_net:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                        lr=args.lr_hid,
                                        weight_decay=args.weight_decay,betas=(args.beta1,0.999))
 
    def forward_test_phrase(self,ir_train,trainImageSize):
        img_hig = np.zeros((1,1,trainImageSize,trainImageSize), dtype = np.float32)
        img_hig[0,0,:,:]=np.copy(ir_train)
        self.set_input_test(img_hig)
        self.forward()
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
    def set_input_train(self,img_s,img_t,img_icvl):
        img_s=torch.FloatTensor(img_s)
        img_t=torch.FloatTensor(img_t)
        
        self.real_A = img_s.to(self.device)
        self.real_B = img_t.to(self.device)
        if self.args.lambda_loss_d_real_icvl>0:
            img_icvl=torch.FloatTensor(img_icvl)
            self.real_C = img_icvl.to(self.device)
        
    def set_input_test(self,img_s):
        img_s=torch.FloatTensor(img_s)
        self.real_A = img_s.to(self.device)
        
    
    
    def forward(self):
        self.fake_B = self.netG(self.real_A)  # G(A)
        
    '''
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        #pred_fake=self.netD(self.fake_B.detach())#gypark:IR->depth (ours)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        #real_AB=self.real_B#gypark: depth(ours/bighand)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        #pred_fake=self.netD(self.fake_B) #gypark:cGAN->GAN
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.args.lambda_L1
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    '''
    
    
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
            real_AB=self.real_B#gypark: depth(ours/bighand)
            pred_real = self.netD(real_AB)
        else:
            real_AB=torch.cat((self.real_A, self.real_B), 1)
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)        
        
        
        # Real (ICVL)
        if self.args.lambda_loss_d_real_icvl>0:
            real_C=self.real_C
            pred_real_C = self.netD(real_C)
            self.loss_D_real_icvl = self.criterionGAN(pred_real_C, True)
            # combine loss and calculate gradients
            self.loss_D = self.loss_D_fake*self.args.lambda_loss_d_fake + self.loss_D_real * self.args.lambda_loss_d_real + (self.loss_D_real_icvl)*self.args.lambda_loss_d_real_icvl
            self.loss_D.backward()
        else:
            # combine loss and calculate gradients
            self.loss_D = self.loss_D_fake*self.args.lambda_loss_d_fake + self.loss_D_real * self.args.lambda_loss_d_real 
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
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) 
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 * self.args.lambda_L1
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

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
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
        
        
        
        
      
    def getloss(self):
        out=[]
        
        val=self.args.lambda_L1 * self.loss_G_L1.item()
        out.append(val)
        
        val=self.loss_G_GAN.item()
        out.append(val)
        
        val=self.args.lambda_loss_d_real * self.loss_D_real.item()
        out.append(val)
        
        val=self.args.lambda_loss_d_fake * self.loss_D_fake.item() 
        out.append(val)
        
        return out

        

if __name__ == '__main__':
    data=np.random.rand(1,1,128,128)
    
    norm='batch'
    norm_layer = get_norm_layer(norm_type=norm)
    
    #generator
    unet=UnetGenerator(1,1,7,64,norm_layer=norm_layer,use_dropout=False)
    
    input=torch.FloatTensor(data)
    output=unet(input)
    print(output.shape)
    
    #discriminator
    dnet=NLayerDiscriminator(1,64,n_layers=3,norm_layer=norm_layer)
    output=dnet(input)
    print(output.shape)
    
    
    
    
    