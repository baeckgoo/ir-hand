
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from numpy.testing import assert_array_almost_equal
import matplotlib.pyplot as plt

from tqdm import tqdm
from tqdm import trange
import pickle
import csv

from datasetLoader_icvl import Utils 
from datasetLoader_icvl import DatasetLoader
import time

        

def loadLabel(label_path,frame):
    csv_file=open(label_path,'r')
    csv_reader=csv.reader(csv_file,delimiter='\t')
    
    i=0
    for line in csv_reader:
        #self.joint3d_gt[frame,:]=line[1:-1]
        if i==frame:
            out=line[1:-1]
            break
        i+=1
            
    csv_file.close()
    return np.asarray(out,'float')
  
def segmentHand(img,joint3d,jointNum,calibMat,camerawidth,cameraheight):
    #self.window3d : pixel,pixel,mm
    ground_3dpos_=np.copy(joint3d)
    ground_3dpos=np.reshape(ground_3dpos_,(jointNum,3))

    calibMat=calibMat
    ground_2dpos=np.matmul(calibMat,np.transpose(ground_3dpos))/ground_3dpos[:,2]
    ground_2dpos=np.transpose(ground_2dpos)
        
    rest=[30,30,50]
    window3d=np.asarray([[ground_2dpos[:,0].min()-rest[0],ground_2dpos[:,0].max()+rest[0]], #x
                             [ground_2dpos[:,1].min()-rest[1],ground_2dpos[:,1].max()+rest[1]], #y
                             [ground_3dpos[:,2].min()-rest[2],ground_3dpos[:,2].max()+rest[2]]]) #z
        
    w=camerawidth
    h=cameraheight
        
    x=[int(window3d[0,0]),int(window3d[0,1])]
    y=[int(window3d[1,0]),int(window3d[1,1])]
    z=[int(window3d[2,0]),int(window3d[2,1])]
    #print(x,y,z)
        
    if x[0]<=-rest[0] or y[0]<=-rest[1] or x[1]>(w-1+rest[0]) or y[1]>(h-1+rest[1]):
        return None,False
        
    #bound
    x[0]=min(max(x[0],0),w-1) 
    y[0]=min(max(y[0],0),w-1)
        
    x[1]=min(max(x[1],0),h-1) 
    y[1]=min(max(y[1],0),h-1) 
    #print(x,y,z)
        
        
    '''TIME consuming'''
    img_seg=img.copy()
    img_seg[:,0:x[0]]=0
    img_seg[:,x[1]:w]=0
    img_seg[0:y[0],:]=0
    img_seg[y[1]:h,:]=0
    img_seg[img_seg<z[0]]=0
    img_seg[img_seg>z[1]]=0
    '''TIME consuming'''
        
    return img_seg,True
    

if __name__=="__main__":   
    ##--user setting
    
    ##--common setting
    outDim=45
    datanum=119629
    augmentation_num=5
    jointNum=21
    
    fx=475.065948
    fy=475.065857
    cx=315.944855
    cy=245.287079
    cube=np.asarray([250,250,250])
    utils=Utils(jointNum,fx,fy,cx,cy,cube)
    
    label_path="/home/yong/hdd/dataset/HANDS17/training/original/Training_Annotation.txt"
    #label_path="/home/yong/hdd/dataset/HANDS17/training/original/Test_Annotation.txt"
    
    image_path="/home/yong/hdd/dataset/HANDS17/training/original/images_train/"
    #image_path="/home/yong/hdd/dataset/HANDS17/training/original/images_test/
    
    
    calibMat=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]]) 
    camerawidth=640
    cameraheight=480
    
    rng= np.random.RandomState(np.random.randint(1,100))
        
    ##--start
        
    #--load a depth image and label
    frame=1
    img=cv2.imread(image_path+"image_D%08d.png"%(frame+1),2)
    #plt.imshow(img)
    
    joints3d=loadLabel(label_path,frame)
    
    #--segment hand image
    TIME=time.time()
    depth_seg,isvalid_bool=segmentHand(img,joints3d,jointNum,calibMat,camerawidth,cameraheight)
    print('seg time..',time.time()-TIME)
    #plt.imshow(depth_seg)
    
    #--preprocess
    com=utils.calculateCOM(depth_seg)
    com,depth_crop,window=utils.refineCOMIterative(depth_seg,com,3)     
    depth_train=utils.makeLearningImage(depth_crop,com)
    
    plt.imshow(depth_train)
    
    #--parameter for augmentation
    sigma_com = 5.
    rot_range=180.
    sigma_sc = 0.02
    
    off=rng.randn(3)*sigma_com # mean:0, var:sigma_com (normal distrib.)
    
    rot=rng.uniform(-rot_range,rot_range)
    rot=np.mod(rot,360)
               
    sc = np.fabs(rng.randn(1) * sigma_sc + 1.)
    
    #?????
    img_seg=depth_seg## caution!
    
    #rotation
    TIME=time.time()
    img_seg,joints3d=utils.augmentRotation(img_seg,joints3d,com,rot)
    #plt.imshow(img_seg)
    print('augmentRotation:',time.time()-TIME)
    #translation
    TIME=time.time()
    com2d,com3d,_=utils.augmentTranslation(img_seg,joints3d,com,off)
    print('augmentTranslation:',time.time()-TIME)
    #scale
    #cube=datasetloader.cube
    TIME=time.time()
    cube,window=utils.augmentScale(img_seg,cube,com2d,sc)
    print('augmentScale:',time.time()-TIME)
    
    #--apply augmentation to image 
    xstart,xend,ystart,yend,zstart,zend=window[0],window[1],window[2],window[3],window[4],window[5]
    img_crop=utils.crop(img_seg,xstart,xend,ystart,yend,zstart,zend)
    img_train=utils.makeLearningImage(img_crop,com)

    
    
    #--apply augmentation to joints
    joints3d_norm=np.concatenate(joints3d)-np.tile(com3d,(1,21))
    joints3d_norm2=joints3d_norm/(cube[2]/2.)
    joints3d_norm3=np.clip(joints3d_norm2,-1,1)
    
    #--show
    cv2.line(img_seg,(window[0],window[2]),(window[1],window[2]),(255,0,0),5)
    cv2.line(img_seg,(window[0],window[2]),(window[0],window[3]),(255,0,0),5)
    cv2.line(img_seg,(window[1],window[2]),(window[1],window[3]),(255,0,0),5)
    cv2.line(img_seg,(window[0],window[3]),(window[1],window[3]),(255,0,0),5)
    #plt.imshow(img_seg)
    
    joints2d=utils.projectAlljoints2imagePlane(joints3d)
    utils.circle2DJoints2Image(img_seg,joints2d)
    plt.imshow(img_seg)
        
        
    
    
    
    
    