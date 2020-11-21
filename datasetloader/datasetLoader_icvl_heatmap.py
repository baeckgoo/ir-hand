import cv2
import numpy as np
import csv
import sys
import time
#from pyquaternion import Quaternion

import random
import os

import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
import time


#import pandas as pd

def nothing(x):
    pass

class Utils:
    def __init__(self,traindataNum,jointNum,fx,fy,cx,cy):
        self.jointNum=jointNum
        self.fx=fx
        self.fy=fy
        self.cx=cx
        self.cy=cy
        self.calibMat=np.asarray([[self.fx,0,self.cx],[0,self.fy,self.cy],[0,0,1]]) 
        self.traindataNum=traindataNum
        
    def projectJoint2imagePlane(self,pos):
        calibMat=self.calibMat
        p2d=np.matmul(calibMat,np.transpose(pos))
        pos2d=p2d/p2d[2,:]
        pos2d=np.transpose(pos2d[0:2,:])
        return pos2d.astype(np.int32)
        
        
    def projectAlljoints2imagePlane(self,ground_3dpos):
        jointNum=self.jointNum
        calibMat=self.calibMat
        pos2d=np.zeros((jointNum,2))
        
        p2d=np.matmul(calibMat,np.transpose(ground_3dpos)) #(3*3)*(N*3)^T
        pos2d=p2d/p2d[2,:]
        pos2d=np.transpose(pos2d[0:2,:])
        return pos2d.astype(np.int32)
        
    def getRefinedCOM(self):
        return self.com_refined.astype(np.float)
    
    def refineCOMIterative(self,dimg,com,num_iter):
        dpt=dimg.copy()
        for k in range(num_iter):
            #size=np.asarray(size)*(1-0.1*k)
            #print(size)
            xstart, xend, ystart, yend, zstart, zend=self.comToBounds(com)
            
            xstart=max(xstart,0)
            ystart=max(ystart,0)
            xend=min(xend,dpt.shape[1])
            yend=min(yend,dpt.shape[0])

            cropped=self.getCrop(dpt,xstart,xend,ystart,yend,zstart,zend)
            
            
            com=self.calculateCOM(cropped)
            
            if np.allclose(com,0.):
                com[2]=cropped[cropped.shape[0]//2,cropped.shape[1]//2]
            com[0]+=max(xstart,0)
            com[1]+=max(ystart,0)
    
        return com,cropped,[xstart,xend,ystart,yend,zstart,zend]

    def getCrop(self,dpt, xstart, xend, ystart, yend, zstart, zend, thresh_z=True):
        cropped = dpt[ystart:yend, xstart:xend].copy()
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = zstart
        cropped[msk2] = 0.
        return cropped

    def comToBounds(self,com):
        fx=self.fx
        fy=self.fy
        size=[250,250,300,250,200,200] 
        
        zstart = com[2] - size[4] / 2.
        zend = com[2] + size[5] / 2.
        xstart = int(np.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
        xend = int(np.floor((com[0] * com[2] / fx + size[1] / 2.) / com[2]*fx))
        ystart = int(np.floor((com[1] * com[2] / fy - size[2] / 2.) / com[2]*fy))
        yend = int(np.floor((com[1] * com[2] / fy + size[3] / 2.) / com[2]*fy))
        
        return xstart, xend, ystart, yend, zstart, zend

    def calculateCOM(self,dimg,minDepth=10,maxDepth=500):
        dc=dimg.copy()
        dc[dc<minDepth]=0
        dc[dc>maxDepth]=0
        cc= ndimage.measurements.center_of_mass(dc>0)
        num=np.count_nonzero(dc)
        com=np.array((cc[1]*num,cc[0]*num,dc.sum()),np.float)
        
        if num==0:
            return np.array((0,0,0,np.float))
        else:
            return com/num
        
    def detectHand(self,dimg):
        maxDepth=500
        minDepth=10
        steps=10
        dz=(maxDepth-minDepth)/float(steps)
    
        maxContourSize=100
        for j in range(steps):
            part=dimg.copy()
            part[part<j*dz+minDepth]=0
            part[part>(j+1)*dz+minDepth]=0
            part[part!=0]=10
                
            ret,thresh=cv2.threshold(part,1,255,cv2.THRESH_BINARY)
            thresh=thresh.astype(dtype=np.uint8)
            contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(dimg3c, contours, -1, (0,255,0), 3)
            
            for c in range(len(contours)):
                if cv2.contourArea(contours[c])>maxContourSize:
                    maxContourSize=cv2.contourArea(contours[c])
                    
                    #centroid
                    M=cv2.moments(contours[c])
                    cx=int(np.rint(M['m10']/M['m00']))
                    cy=int(np.rint(M['m01']/M['m00']))
                        
                    #crop
                    xstart=int(max(cx-100,0))
                    xend=int(min(cx+100,part.shape[1]-1))
                    ystart=int(max(cy-100,0))
                    yend=int(min(cy+100,part.shape[0]-1))
                        
                    cropped=dimg[ystart:yend,xstart:xend].copy()
                    cropped[cropped<j*dz+minDepth]=0
                    cropped[cropped>(j+1)*dz+minDepth]=0
                    
                    part_selected=part.copy()
            
        if maxContourSize==100:
            return part,[0,0,0,0]
        else:      
            return cropped,[xstart,xend,ystart,yend]
        
    def _relative_joints(self,width,height,ground_2dpos,window3d):
        ground_2dpos_crop= ground_2dpos-[window3d[0,0],window3d[1,0]]
        x=ground_2dpos_crop[:,0]*width/(window3d[0,1]-window3d[0,0])
        y=ground_2dpos_crop[:,1]*height/(window3d[1,1]-window3d[1,0])
        return np.transpose(np.asarray([x,y],'int32')) 
    
    def _generate_hm(self, height, width ,joints, maxlenght):
        num_joints = joints.shape[0]
        hm = np.zeros((num_joints,height, width), dtype = np.float32)
        for i in range(num_joints):
            if not(np.array_equal(joints[i], [-1,-1])):
                s = int(np.sqrt(maxlenght) * maxlenght * 10 / (width*height)) + 2
                hm[i,:,:] = self._makeGaussian(height, width, sigma= s, center= (joints[i,0], joints[i,1]))
            else:
                hm[i,:,:] = np.zeros((height,width))
        return hm      				
		       
    def _makeGaussian(self, height, width, sigma = 3, center=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 =  width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0= center[1]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2) 
    
class DatasetLoader:
    def __init__(self,dataset_path):
        self.dataset_path=dataset_path
        fx=475.065948
        fy=475.065857
        cx=315.944855
        cy=245.287079
        self.calibMat=np.asarray([[fx,0,cx],[0,fy,cy],[0,0,1]])   
        self.jointNum=21
        
        self.camerawidth=640
        self.cameraheight=480
    
        self.traindataNum=95703*4 #957032
        self.validateNum=95703*4
        self.traindataNum_full=957032
        self.trainImageSize=256
        self.htmapSize=64
        
        self.utils=Utils(self.traindataNum,self.jointNum,fx,fy,cx,cy)  
         
        self.trainlabel=np.zeros((self.traindataNum_full,self.jointNum*3))
        #self.trainlabel=np.asarray(self.trainlabel,'float32')
        self.loadFullLabels()
        
        self.idx_train=np.arange(self.traindataNum)
        self.idx_validate=np.arange(self.traindataNum,self.traindataNum+self.validateNum)
        
        
    def generator_learningData(self,batch_size,opt):
        if opt=='train':
            np.random.shuffle(self.idx_train)
        elif opt=='validate':
            np.random.shuffle(self.idx_validate)
            
        #print(self.idx_train)
        trainImageSize=self.trainImageSize
        htmapSize=self.htmapSize
        jointNum=self.jointNum
        train_img = np.zeros((batch_size,1,trainImageSize,trainImageSize), dtype = np.float32)
        train_htmap = np.zeros((batch_size,jointNum,htmapSize,htmapSize), np.float32)
        
        j=0
        while True:
            
            #for i in range(batch_size):
            i=0
            while True:
                if opt=='train':
                    if j==self.traindataNum:
                        j=0
                elif opt=='validate':
                    if j==self.validateNum:
                        j=0
                    
                if opt=='train':
                    frame=self.idx_train[j]
                elif opt=='validate':
                    frame=self.idx_validate[j]
                
                self.frame=frame
                
                #learning image   
                self.makeBoundBox(frame)
                if self.isValidDataFrame()==False:
                    j+=1
                    #print('false..',frame)
                    continue

                img_crop=self.makeLearningImage(frame)  #0.003 sec  
                train_img[i,0,:,:]=img_crop.copy()
                
                #heatmap
                ground_3dpos_=np.copy(self.trainlabel[frame,:])
                ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))            
                ground_2dpos=self.utils.projectAlljoints2imagePlane(ground_3dpos)
                                
                window2d=self.getBoundBox()
                new_j=self.utils._relative_joints(htmapSize,htmapSize,ground_2dpos,window2d)
                hm = self.utils._generate_hm(htmapSize, htmapSize, new_j, trainImageSize)
                train_htmap[i]=hm.copy()
                
                #augmentation
                j+=1
                i+=1
                if i==batch_size:
                    break
                       
            yield train_img, train_htmap
               
    
    def isValidDataFrame(self):
        window3d=self.getBoundBox()
        cw=self.camerawidth
        ch=self.cameraheight
        #print('window3d',window3d)
        #x limit
        if (window3d[0,0]<=0 or window3d[0,1]<=0) or (window3d[1,0]<=0 or window3d[1,1]<=0):   
            return False
        #y limit
        if (window3d[0,0]>=cw or window3d[0,1]>=cw) or (window3d[0,0]>=ch or window3d[0,1]>=ch):
            return False
        return True
    
    def getBoundBox(self):
        return np.copy(self.window3d)
    
    def makeBoundBox(self,frame):  
        #print(self.trainlabel[frame,:])
        ground_3dpos_=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        points_palm=ground_3dpos[[0,1,2,3,4,5],:]
    
        calibMat=self.calibMat
        ground_2dpos=np.matmul(calibMat,np.transpose(ground_3dpos))/ground_3dpos[:,2]
        ground_2dpos=np.transpose(ground_2dpos)
        
        rest=[20,20,70]
        window3d=np.asarray([[ground_2dpos[:,0].min()-rest[0],ground_2dpos[:,0].max()+rest[0]], #x
                             [ground_2dpos[:,1].min()-rest[1],ground_2dpos[:,1].max()+rest[1]], #y
                             [ground_3dpos[:,2].min()-rest[2],ground_3dpos[:,2].max()+rest[2]]]) #z
    
        self.window3d=np.asarray(window3d,'int')
        
    
        #window3d=np.asarray(window3d,'int')
        #return window3d
    
    def makeBoundBoxFromCom(self,com):
        calibMat=self.calibMat
        x0=100
        y0=120
        z0=150
        x1=150
        y1=150
        z1=80
        boxrange=[[-x0,-y0,-z0],[x1,-y0,-z0],[-x0,y1,-z0],[x1,y1,-z0], #front of box
                  [-x0,-y0,z1],[x1,-y0,z1],[-x0,y1,z1],[x1,y1,z1]] #back of box
        window3d=com+boxrange
        window2d=np.matmul(calibMat,np.transpose(window3d))/window3d[:,2]
        
        window2d[0,window2d[0,:]<0]=0
        window2d[0,window2d[0,:]>self.camerawidth-1]=self.camerawidth-1
        window2d[1,window2d[1,:]<0]=0
        window2d[1,window2d[1,:]>self.cameraheight-1]=self.cameraheight-1       
        window2d=np.asarray(window2d,'int')
        
        return window3d,np.transpose(window2d)
      
    def segmentHand(self,img,window2d):
        w=self.camerawidth
        h=self.cameraheight
        
        x=[int(window2d[0,0]),int(window2d[0,1])]
        y=[int(window2d[1,0]),int(window2d[1,1])]
        z=[int(window2d[2,0]),int(window2d[2,1])]
        
        img_seg=img.copy()
        img_seg[:,0:x[0]]=0
        img_seg[:,x[1]:w]=0
        img_seg[0:y[0],:]=0
        img_seg[y[1]:h,:]=0
        img_seg[img_seg<z[0]]=0
        img_seg[img_seg>z[1]]=0
        
        return img_seg
    
    def cropHand(self,img,window2d):
        img_crop=img.copy()
        x=[window2d[0,0],window2d[0,1]]
        y=[window2d[1,0],window2d[1,1]]
        
        img_crop=img_crop[y[0]:y[1],x[0]:x[1]]
        img_crop=np.asarray(img_crop,'float32')
        
        return img_crop

    def makeTrackingImage(self,img,frame):
        ground_3dpos_=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        points_palm=ground_3dpos[[0,1,2,3,4,5],:]
        
        #bounding box & bounding window
        #com=np.mean(points_palm,0)
        com=points_palm[2]
        #window3d=self.makeBoundBox(frame)
        self.makeBoundBox(frame)
        window3d=self.window3d

        #segment and crop image from the box
        img_seg=self.segmentHand(img,window3d)
        
        dimg_track=cv2.resize(img_seg,(320,240))
        dimg_track=np.asarray(dimg_track,'float32')
       
        return dimg_track
    
    def showBox(self,frame):
        img=self.loadImage(frame)
        ground_3dpos_=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        points_palm=ground_3dpos[[0,1,2,3,4,5],:]
        
        com=points_palm[2]
        window3d=self.makeBoundBox(frame)
        
        p0=(window3d[0,0],window3d[0,1])
        p1=(window3d[1,0],window3d[1,1])
        p2=(window3d[2,0],window3d[2,1])
        p3=(window3d[3,0],window3d[3,1])
        cv2.line(img,p0,p1,(255,0,0),5)
        cv2.line(img,p0,p2,(255,0,0),5)
        cv2.line(img,p1,p3,(255,0,0),5)
        cv2.line(img,p2,p3,(255,0,0),5)
        cv2.imshow('box',np.uint8(img))
        cv2.waitKey(1)
     
    def limitBoundBox(self):
        #limit
        if self.window3d[0,0]<=0:
            self.window3d[0,0]=0
        if self.window3d[0,1]>=self.camerawidth-1:
            self.window3d[0,1]=self.camerawidth-1
        if self.window3d[1,0]<=0:
            self.window3d[1,0]=0
        if self.window3d[1,1]>=self.cameraheight-1:
            self.window3d[1,1]=self.cameraheight-1
            
    #should run 'makeBoundBox(frame)' function before. 
    def makeLearningImage(self,frame):
        img=self.loadImage(frame) #0.002 sec
        ground_3dpos_=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        points_palm=ground_3dpos[[0,1,2,3,4,5],:]
        
        #bounding box & bounding window
        #com=np.mean(points_palm,0)
        com=points_palm[2]
        self.limitBoundBox()
        window3d=self.getBoundBox()
        
        
        #segment, crop image, and normalize from the box
        img_seg=self.segmentHand(img,window3d) #0.0003 sec
        img_crop=self.cropHand(img_seg,window3d)
        
        s=self.trainImageSize
        img_crop[img_crop==0]=np.max(img_crop)
        d_max=np.max(img_crop)
        d_min=np.min(img_crop)
        if d_max-d_min<1:
            print('dmax-dmin<! frame%d min%f max:%f'%(frame,d_min,d_max))
        img_crop=-1+2*(img_crop-d_min)/(d_max-d_min)        
        #return img_seg,cv2.resize(img_crop,(s,s))
        
        cnnimg=cv2.resize(img_crop,(s,s))

        return np.copy(cnnimg)
    
    def makeHeatMap(self,frame):
        ground_3dpos_=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(self.jointNum,3))
        ground_2dpos=self.utils.projectAlljoints2imagePlane(ground_3dpos)
        
        #window2d=datasetloader.window2d
        window3d=self.getBoundBox()
        s=self.htmapSize
        new_j=self.utils._relative_joints(s,s,ground_2dpos,window3d)
        self.new_j=new_j
        hm = self.utils._generate_hm(s, s, new_j, s)
        return hm
    
    def splitDataset(self,kfold,k_idx):
        ratio=self.traindataNum//kfold
        
        #full dataset
        mylist=np.arange(self.traindataNum)
        #validation dataset
        self.idx_valid = mylist[k_idx*ratio:(k_idx+1)*ratio]
        #train dataset
        if k_idx==0:
            self.idx_train = mylist[(k_idx+1)*ratio:]
        else:
            idx_train_0=mylist[0:k_idx*ratio]
            idx_train_1=mylist[(k_idx+1)*ratio:]
            self.idx_train=np.concatenate((idx_train_0,idx_train_1))
                    
    def loadImage(self,frame):
        #image in ground truth
        image_path=self.dataset_path
        image_fname=image_path+"images/image_D%08d.png"%(frame+1)
        img=cv2.imread(image_fname,2)

        return img

    def loadFullLabels(self):
        path=self.dataset_path
        label_fname=path+"Training_Annotation.txt"
    
        csv_file=open(label_fname,'r')
        csv_reader=csv.reader(csv_file,delimiter='\t')
    
        frame=0
        for line in csv_reader:
            self.trainlabel[frame,:]=line[1:-1]
            frame+=1
            if frame==self.traindataNum_full:
                break
            
        csv_file.close()

    def showLabel(self,frame,img):
        jointNum=self.jointNum
        calibMat=self.calibMat
        
        ground_3dpos=np.copy(self.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos,(jointNum,3))
        
        img2=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

        color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255]]
        for i in range(jointNum):
            pos3d=np.asarray([ground_3dpos[i,0],ground_3dpos[i,1],ground_3dpos[i,2]])
            pos=np.matmul(calibMat,pos3d)/pos3d[2]

            if i<6:
                cv2.circle(img2,(int(pos[0]),int(pos[1])),5,(255,0,255),-1)
            else:
                colorid=int((i-6)/3)
                cv2.circle(img2,(int(pos[0]),int(pos[1])),5,color[colorid],-1)
    
        cv2.imshow("label",np.uint8(img2))
        cv2.waitKey(1)
    
 
def run_modeltracking(dimg_track,renderer,handtracker,handPose,wname,frame):
    renderer.transferObservation2GPU(dimg_track)
    
    for i in range(50):        
        handPose=handtracker.run_original(handPose,{})
        
        
    #render the hand model    
    dimg3c_norm = cv2.normalize(dimg_track, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)       
    dimg3c_norm=np.uint8(cv2.cvtColor(dimg3c_norm,cv2.COLOR_GRAY2BGR)) 
    
    handPose_render=np.copy(handPose)
    renderer.render(handPose_render,b'color')
    model_img=renderer.getResizedColorTexture(width,height)        
    final_img=cv2.addWeighted(model_img,0.6,dimg3c_norm,0.6,0)
    cv2.imshow(wname,final_img)
    cv2.imwrite("bighand_wrist/%d.png"%frame,final_img)
    cv2.waitKey(1)

    f = open('rotation_ref.txt', 'w', encoding='utf-8', newline='')
    wr = csv.writer(f)
    wr.writerow(handPose[3:7])
    f.close() 
    
    return handPose
    
def calculateRotation(a,b):
    #calculate rotation from reference frame.
    avg_a=np.mean(a,0)
    avg_b=np.mean(b,0)
    
    A=np.mat(a-avg_a)
    B=np.mat(b-avg_b)
    
    H=np.transpose(A)*B #  ref-> target  or target->ref
    #H=np.transpose(B)*A #  ref-> target  or target->ref
    
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T*U.T
    print('R',R)
    # special reflection case
    if np.linalg.det(R) < 0:
        print( "Reflection detected")
        Vt[2,:] *= -1
        R = Vt.T * U.T
    
    return Quaternion(matrix=R)  

def show3Dposition(img,position3d,jointNum,calibMat):
        
    img2=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    color=[[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,255,255],[0,255,255]]
    for i in range(jointNum):
        pos3d=np.asarray([position3d[i,0],position3d[i,1],position3d[i,2]])
        pos=np.matmul(calibMat,pos3d)/pos3d[2]
        cv2.circle(img2,(int(pos[0]),int(pos[1])),5,color[i],-1)
        
    cv2.imshow("rot",np.uint8(img2))
    cv2.waitKey(1)
    
    

#def test_internalcode():
if __name__=="__main__":
    frame=95703*2-1  #12435,  91040, 62505, 42275, 87439  , 20545

    #make quaternion data
    #setting
    jointNum=21
    width=int(640)
    height=int(480)
    trainImageSize=256
    htmapSize=256
    batchsize=1
    #test to make training dataset
    
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    #shuffler=datasetloader.shuffleGenerator()
    
    batch_size=16
    stacks=1
    joints_num=21
    
    while True:
        #load
        #frame=next(shuffler)
        img=datasetloader.loadImage(frame)
        ground_3dpos_=np.copy(datasetloader.trainlabel[frame,:])
        ground_3dpos=np.reshape(ground_3dpos_,(datasetloader.jointNum,3))
        #print('gt_3dpos',ground_3dpos)
        plt.imshow(img)
      
        
        #learning image
        datasetloader.makeBoundBox(frame)
        if datasetloader.isValidDataFrame()==False:
            print("not valid data frame")
            break
        img_crop=datasetloader.makeLearningImage(frame)
            
        #debug cropped image
        train_img_vis=cv2.normalize(img_crop, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            
        #heatmap
        ground_2dpos=datasetloader.utils.projectAlljoints2imagePlane(ground_3dpos)
        window2d=datasetloader.getBoundBox()
        new_j=datasetloader.utils._relative_joints(htmapSize,htmapSize,ground_2dpos,window2d)
        hm = datasetloader.utils._generate_hm(htmapSize, htmapSize, new_j, trainImageSize)
        
        #visualize original image (ok) 
        for j in range(joints_num):
            cv2.circle(img,(ground_2dpos[j,0],ground_2dpos[j,1]),5,(255,255,0),-1)
        cv2.line(img,(window2d[0,0],window2d[1,0]),(window2d[0,1],window2d[1,0]),(255,0,0),5)
        cv2.line(img,(window2d[0,0],window2d[1,0]),(window2d[0,0],window2d[1,1]),(255,0,0),5)
        cv2.line(img,(window2d[0,1],window2d[1,0]),(window2d[0,1],window2d[1,1]),(255,0,0),5)
        cv2.line(img,(window2d[0,0],window2d[1,0]),(window2d[0,1],window2d[1,0]),(255,0,0),5)
        #plt.imshow(img)
           
        #visualize heat map (ok)
        train_img_vis=cv2.normalize(img_crop, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        train_img_vis3c=cv2.applyColorMap(np.uint8(train_img_vis),1)
            
        for j in range(joints_num):
            _hm=cv2.resize(hm[:,:,j],(trainImageSize,trainImageSize))
            _hm=np.uint8(cv2.normalize(_hm, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
            if j==0:
                _hm2=np.zeros(_hm.shape,np.uint8)
            _hm2=cv2.addWeighted(_hm2,1.0,_hm,1.0,0)
                    
        _hm_vis=cv2.applyColorMap(np.uint8(_hm2),1)
        final_hm=cv2.addWeighted(train_img_vis3c,0.8,_hm_vis,1.0,0)
        #plt.imshow(final_hm)

        break
    
#if __name__=="__main__":
def test_generator_learningData():
    frame=0##7

    #make quaternion data
    #setting
    jointNum=21
    width=int(640)
    height=int(480)
    trainImageSize=256
    htmapSize=64
    batchsize=32
    #test to make training dataset
    
    datasetloader=DatasetLoader("../../../dataset/HANDS17/training/")
    #shuffler=datasetloader.shuffleGenerator()
    
    batch_size=16
    joints_num=21
    
    for epoch in range(1):  
        print('epoch..',epoch)
        generator=datasetloader.generator_learningData(batch_size,'train')
                   
        iter_num_train=len(datasetloader.idx_train)//batch_size
        for i in range(iter_num_train):
            start_time=time.time()
            img_train,gt_train=next(generator)   
            print('time:',time.time()-start_time)
            break
            """
            #cv2.imshow('img_train',img_train[0,0,:,:])
            gt_train_debug=np.zeros((htmapSize,htmapSize))
            for j in range(joints_num):
                gt_train_=gt_train[0,j,:,:].copy()
                maxval=np.max(gt_train_)
                gt_train_[gt_train_<maxval*0.9]=0
                gt_train_debug+=gt_train_
            """
        #cv2.imshow("gt_train",gt_train_debug)
            
      
        
    cv2.waitKey(1)
        
        
    #test shuffle index
    """
    if not 'datasetloader' in locals():
        datasetloader=DatasetLoader()
    
        
    epoch=5
    batch_size=5
    iterNum=datasetloader.traindataNum//batch_size
    print('iteration num',iterNum)
    for epoc in range(epoch):
        print('')
        print('')
        print('epoch',epoc)
        datasetloader.shuffleTrainDataIndex()
        generator=datasetloader._aux_generator_3Djoints(batch_size)
        
        for i in range(iterNum):
            a,b=next(generator)
            cv2.imshow('a',np.uint8(a[0,:,:,0]))
            cv2.waitKey(1000)
            print('kkk',i)
           
        #generator= datasetloader._aux_generator_3Djoints(frame)

    """
    

        
    
    
    
    
    
  
            
            
    
    
    
    
    
    
        
    
    
    



















