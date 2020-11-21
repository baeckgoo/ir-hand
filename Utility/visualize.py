import cv2
import numpy as np


def show_joints():
    a=0
    

class Visualize_combined_outputs():
    def __init__(self,utils,xnum,ynum,camerawidth,cameraheight):
        '''
        xnum: the nubmer of window (x axis)
        ynum: the number of window (y axis)
        xs: the window size (width)
        ys: the window size (height)
        '''
        
        self.img_final=np.zeros((cameraheight*ynum,camerawidth*xnum,3),'uint8')
        self.utils=utils
        self.container={}
       
        self.camerawidth=camerawidth
        self.cameraheight=cameraheight
        
    def setWindow(self,name,window_idx):
        self.container[name]={}
        self.container[name]['output_img']=[]
        
        self.container[name]['input_imgs']=[]
        self.container[name]['idx']=window_idx
        
        
   
        
    def set_inputimg(self,name,img):
        self.container[name]['input_imgs'].append(img)
        
    def set_outputimg(self,name,img):
        self.container[name]['output_img']=img
 
    def convertTo3cInputImages(self,name):
        input_imgs=self.container[name]['input_imgs']
            
        for i,img in enumerate(input_imgs):
            size=np.shape(img)[0]
            #self.container[name]['input_imgs'][i]=cv2.cvtColor(np.uint8(np.clip(size*(img+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
            self.container[name]['input_imgs'][i]=cv2.cvtColor(np.uint8(np.clip(size*(img+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
                
    '''
    def convertTo3cInputImages(self):
        for kk,name in enumerate(self.container):
            input_imgs=self.container[name]['input_imgs']
            
            for i,img in enumerate(input_imgs):
                size=np.shape(img)[0]
                #self.container[name]['input_imgs'][i]=cv2.cvtColor(np.uint8(np.clip(size*(img+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
                self.container[name]['input_imgs'][i]=cv2.cvtColor(np.uint8(np.clip(size*(img+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
   
    '''
    
    def set_finalimg(self,name,output_recon):    
        output_img=self.container[name]['output_img']
        #outimg3c=np.uint8(cv2.normalize(output_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
        outimg3c=output_img
        outimg3c=cv2.cvtColor(outimg3c,cv2.COLOR_GRAY2BGR)
        outimg3c=self.utils.circle3DJoints2Image(outimg3c,output_recon[0,:])   
        
        input_imgs=self.container[name]['input_imgs']
        
        for i,input_img in enumerate(input_imgs):
            outimg3c[0:128,128*i:128*(i+1),:]=input_img
            
        pos=self.container[name]['idx']    
        self.img_final[self.cameraheight*pos[0]:self.cameraheight*(pos[0]+1),self.camerawidth*pos[1]:self.camerawidth*(pos[1]+1),:]=outimg3c
            
    def get_img_save(self):
        return self.img_final.copy()
        
            
        
class Visualize_hig_hpe1_hpe2():
    def __init__(self,utils):
        self.img_save=np.zeros((480,640*3,3),'uint8')
        self.utils=utils
        
    def set_depth_train(self,depth_train):
        self.depth_train=depth_train.copy()
        
    def set_depth_pred(self,depth_pred):
        self.depth_pred=depth_pred.copy()  
    
    def set_ir_train_hig(self,ir_train):
        self.ir_train_hig=ir_train.copy()
        
    def set_ir_train_hpe2(self,ir_train):
        self.ir_train_hpe2=ir_train.copy()
        
        
    def get_depth_train(self):
        return self.depth_train.copy()  
    
    def get_depth_pred(self):
        return self.depth_pred.copy()
    
    
    def convertTo3cImage(self,trainImageSize):
        depth_train=self.depth_train.copy()
        depth_pred=self.depth_pred.copy()
        ir_train_hig=self.ir_train_hig.copy()
        ir_train_hpe2=self.ir_train_hpe2.copy()
        
        # color-coded depth images (original/predicted)
        '''
        ir3c=cv2.cvtColor(np.uint8(128*(ir_train+1)),cv2.COLOR_GRAY2BGR)
        depth_orig_3c=cv2.applyColorMap(np.uint8(trainImageSize*(depth_train+1)),cv2.COLORMAP_JET)
        depth_pred_3c=cv2.applyColorMap(np.uint8(trainImageSize*(depth_pred+1)),cv2.COLORMAP_JET)
        '''
        
        ir3c_hig=cv2.cvtColor(np.uint8(np.clip(128*(ir_train_hig+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
        ir3c_hpe2=cv2.cvtColor(np.uint8(np.clip(128*(ir_train_hpe2+1)-1,0,255)),cv2.COLOR_GRAY2BGR)
        depth_orig_3c=cv2.applyColorMap(np.uint8(np.clip(trainImageSize*(depth_train+1),0,255)),cv2.COLORMAP_JET)
        depth_pred_3c=cv2.applyColorMap(np.uint8(np.clip(trainImageSize*(depth_pred+1),0,255)),cv2.COLORMAP_JET)
        
        self.ir3c_hig=ir3c_hig
        self.ir3c_hpe2=ir3c_hpe2
        self.depth_orig_3c=depth_orig_3c
        self.depth_pred_3c=depth_pred_3c
        

    
    def calculateDepthError(self,depth_crop):
        depth_train=self.depth_train
        depth_pred=self.depth_pred        
        
        #error between depth images (original/predicted by pix2pix)
        d_min,d_max=np.min(depth_crop),np.max(depth_crop)
        d_recon_pred=d_min+(d_max-d_min)*(depth_pred+1)/2
        d_recon_gt=d_min+(d_max-d_min)*(depth_train+1)/2
                
        mask=d_recon_gt!=np.max(d_recon_gt)
        dif_abs=abs(d_recon_gt[mask]-d_recon_pred[mask])
        #dif_abs=abs(d_recon_gt-d_recon_pred)
        depth_error=np.sum(dif_abs)/np.count_nonzero(mask)
            
        return depth_error
    
    def set_img_hpe1(self,depth_orig,output_recon):
        img_show=np.uint8(cv2.normalize((depth_orig), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
        img_show=self.utils.circle3DJoints2Image(img_show,output_recon[0,:])   
        img_show[0:128,0:128,:]=self.depth_orig_3c
        
        self.img_save[:,0:640]=img_show
        
    def set_img_hig_hpe1(self,ir_orig,output_recon):
        img_show=np.uint8(ir_orig) 
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
        img_show=self.utils.circle3DJoints2Image(img_show,output_recon[0,:])   
        
        img_show[0:128,0:128,:]=self.ir3c_hig
        img_show[128:128*2,0:128,:]=self.depth_pred_3c
        self.img_save[:,640:640*2]=img_show      
        
    def set_img_hpe2(self,ir_orig,output_recon):
        img_show=np.uint8(ir_orig) 
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
        img_show=self.utils.circle3DJoints2Image(img_show,output_recon[0,:])   
                
        img_show[0:128,0:128,:]=self.ir3c_hpe2
        self.img_save[:,640*2:640*3]=img_show    
         
    def resize(self,w,h):
        self.img_save=cv2.resize(self.img_save,(w,h))
        
    def get_img_save(self):
         return self.img_save.copy()
    
    def putText(self,depth_error,err_joints1,err_joints2):
        cv2.putText(self.img_save,'depth error: %.2fmm'%depth_error,(780,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(self.img_save,'consistency error: %.2fmm'%np.mean(err_joints1),(780,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(self.img_save,'consistency error: %.2fmm'%np.mean(err_joints2),(1450,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            
    
class Visualize():
    def __init__(self,utils):
        self.img_save=np.zeros((480,640*2,3),'uint8')
        self.utils=utils
        
    def set_depth_train(self,depth_train):
        self.depth_train=depth_train.copy()
        
    def set_depth_pred(self,depth_pred):
        self.depth_pred=depth_pred.copy()  
        
    def get_depth_train(self):
        return self.depth_train.copy()  
    
    def get_depth_pred(self):
        return self.depth_pred.copy()
    
    def convertTo3cImage(self,ir_train,trainImageSize):
        depth_train=self.depth_train.copy()
        depth_pred=self.depth_pred
        
        # color-coded depth images (original/predicted)
        '''
        ir3c=cv2.cvtColor(np.uint8(128*(ir_train+1)),cv2.COLOR_GRAY2BGR)
        depth_orig_3c=cv2.applyColorMap(np.uint8(trainImageSize*(depth_train+1)),cv2.COLORMAP_JET)
        depth_pred_3c=cv2.applyColorMap(np.uint8(trainImageSize*(depth_pred+1)),cv2.COLORMAP_JET)
        '''
        ir3c=cv2.cvtColor(np.uint8(np.clip(128*(ir_train+1),0,255)),cv2.COLOR_GRAY2BGR)
        depth_orig_3c=cv2.applyColorMap(np.uint8(np.clip(trainImageSize*(depth_train+1),0,255)),cv2.COLORMAP_JET)
        depth_pred_3c=cv2.applyColorMap(np.uint8(np.clip(trainImageSize*(depth_pred+1),0,255)),cv2.COLORMAP_JET)
        
        self.ir3c=ir3c
        self.depth_orig_3c=depth_orig_3c
        self.depth_pred_3c=depth_pred_3c
        
        return ir3c
    
    def calculateDepthError(self,depth_crop):
        depth_train=self.depth_train
        depth_pred=self.depth_pred        
        
        #error between depth images (original/predicted by pix2pix)
        d_min,d_max=np.min(depth_crop),np.max(depth_crop)
        d_recon_pred=d_min+(d_max-d_min)*(depth_pred+1)/2
        d_recon_gt=d_min+(d_max-d_min)*(depth_train+1)/2
                
        mask=d_recon_gt!=np.max(d_recon_gt)
        dif_abs=abs(d_recon_gt[mask]-d_recon_pred[mask])
        #dif_abs=abs(d_recon_gt-d_recon_pred)
        depth_error=np.sum(dif_abs)/np.count_nonzero(mask)
            
        return depth_error
    
    def set_img_save_leftwindow(self,depth_orig,output_recon):
        img_show=np.uint8(cv2.normalize((depth_orig), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))           
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
        img_show=self.utils.circle3DJoints2Image(img_show,output_recon[0,:])   
        img_show[0:128,0:128,:]=self.depth_orig_3c
        
        
        self.img_save[:,0:640]=img_show
        
    def set_img_save_rightwindow(self,depth_orig,output_recon):
        img_show=np.uint8(cv2.normalize((depth_orig), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)) 
        img_show=cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
        img_show=self.utils.circle3DJoints2Image(img_show,output_recon[0,:])   
                
        img_show[0:128,0:128,:]=self.ir3c
        img_show[128:128*2,0:128,:]=self.depth_pred_3c
        self.img_save[:,640:640*2]=img_show           
         
    def get_img_save(self):
         return self.img_save.copy()
    
    def putText(self,depth_error,err_joints):
        cv2.putText(self.img_save,'depth error: %.2fmm'%depth_error,(780,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        cv2.putText(self.img_save,'joints error: %.2fmm'%np.mean(err_joints),(780,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
            
    

    
        