import numpy as np
import cv2

#using blueband
class Segmentation():
    def __init__(self,d_maximum,lower_hsv,upper_hsv,ir_range):
        self.d_maximum=d_maximum
        self.lower_hsv=lower_hsv
        self.upper_hsv=upper_hsv
        self.ir_range=ir_range
   
    def setImages(self,rgb,depth,ir):
        self.rgb=np.copy(rgb)
        self.depth=np.copy(depth)
        self.ir=np.copy(ir)
                
    def makeMask_band(self):
        hsv=cv2.cvtColor(self.rgb,cv2.COLOR_BGR2HSV)
        h_mask=cv2.inRange(hsv,self.lower_hsv,self.upper_hsv)
        h_mask[self.depth>self.d_maximum]=0
                
        contour_mask=np.zeros((h_mask.shape[:2]),dtype='uint8')
        contours, hierachy = cv2.findContours(h_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours)==0:
            #print('makeMask_band: no contour')
            return None
        
        areas=[cv2.contourArea(c) for c in contours]                
        indice=np.argsort(areas)
        c=contours[indice[-1]]
        cv2.drawContours(contour_mask,[c],-1,255,-1)
        h_mask=contour_mask
                
        self.h_mask=cv2.dilate(h_mask,np.ones((7,7),np.uint8),iterations=10) #if not good, increase the iteration number.
        #self.h_mask=cv2.dilate(h_mask,np.ones((7,7),np.uint8),iterations=5)
        
        return 0

    def segmentDepthTwoHands(self,depth):
        depth_seg=depth.copy()
        depth_seg[depth_seg>self.d_maximum]=0
        
        #cv2.imshow('h_mask depth',depth_seg)
        depth_seg_8u=np.uint8(depth_seg)
        depth_seg_8u=cv2.dilate(depth_seg_8u,np.ones((3,3),np.uint8),iterations=1)
        contours, hierachy = cv2.findContours(depth_seg_8u, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
        if len(contours)<2:
            print('segmentDepth: less than 2 (Contour number)')
            return None,None
       
        contour_mask_right=np.zeros((depth_seg_8u.shape[:2]),dtype='uint8')
        contour_mask_left=np.zeros((depth_seg_8u.shape[:2]),dtype='uint8')
        
        areas=[cv2.contourArea(c) for c in contours] 
        indice=np.argsort(areas)
                
        contour_pos1=np.mean(contours[indice[-1]],0)[0] #the first largest
        contour_pos2=np.mean(contours[indice[-2]],0)[0] #the second largest
        
        if contour_pos1[0]<contour_pos2[0]:
            c=contours[indice[-1]]
            cv2.drawContours(contour_mask_right,[c],-1,255,-1)
            c=contours[indice[-2]]
            cv2.drawContours(contour_mask_left,[c],-1,255,-1)
        else:
            c=contours[indice[-1]]
            cv2.drawContours(contour_mask_left,[c],-1,255,-1)
            c=contours[indice[-2]]
            cv2.drawContours(contour_mask_right,[c],-1,255,-1)
            
                
        hand_mask_R=cv2.bitwise_and(depth_seg_8u,depth_seg_8u,mask=contour_mask_right)
        hand_mask_L=cv2.bitwise_and(depth_seg_8u,depth_seg_8u,mask=contour_mask_left)
        
        depth_seg_R=depth_seg.copy()
        depth_seg_L=depth_seg.copy()
        
        depth_seg_R[hand_mask_R==0]=0
        depth_seg_L[hand_mask_L==0]=0
                
        return depth_seg_R,depth_seg_L
    

    def segmentDepthTwoHandsFromband(self):
        depth_seg=self.depth.copy()
        depth_seg[depth_seg>self.d_maximum]=0
        depth_seg[self.h_mask>0]=0
        cv2.imshow('h_mask depth',depth_seg)
        depth_seg_8u=np.uint8(depth_seg)
        depth_seg_8u=cv2.dilate(depth_seg_8u,np.ones((3,3),np.uint8),iterations=1)
        contours, hierachy = cv2.findContours(depth_seg_8u, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
        if len(contours)<2:
            print('segmentDepth: less than 2 (Contour number)')
            return None
       
        contour_mask_right=np.zeros((depth_seg_8u.shape[:2]),dtype='uint8')
        contour_mask_left=np.zeros((depth_seg_8u.shape[:2]),dtype='uint8')
        
        areas=[cv2.contourArea(c) for c in contours] 
        indice=np.argsort(areas)
                
        contour_pos1=np.mean(contours[indice[-1]],0)[0] #the first largest
        contour_pos2=np.mean(contours[indice[-2]],0)[0] #the second largest
        
        if contour_pos1[0]<contour_pos2[0]:
            c=contours[indice[-1]]
            cv2.drawContours(contour_mask_right,[c],-1,255,-1)
            c=contours[indice[-2]]
            cv2.drawContours(contour_mask_left,[c],-1,255,-1)
        else:
            c=contours[indice[-1]]
            cv2.drawContours(contour_mask_left,[c],-1,255,-1)
            c=contours[indice[-2]]
            cv2.drawContours(contour_mask_right,[c],-1,255,-1)
            
        cv2.imshow('right',contour_mask_right)
        cv2.imshow('left',contour_mask_left)
                
        hand_seg_d=cv2.bitwise_and(depth_seg_8u,depth_seg_8u,mask=contour_mask_right)
        depth_seg[hand_seg_d==0]=0
                
        return depth_seg.copy()

    def segmentDepthFromBand(self):
        depth_seg=self.depth.copy()
        depth_seg[depth_seg>self.d_maximum]=0
        depth_seg[self.h_mask>0]=0
         
        depth_seg_8u=np.uint8(depth_seg)
        depth_seg_8u=cv2.dilate(depth_seg_8u,np.ones((3,3),np.uint8),iterations=1)
        contours, hierachy = cv2.findContours(depth_seg_8u, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
        if len(contours)<2:
            print('segmentDepth: less than 2 (Contour number)')
            return None
       
        contour_mask=np.zeros((depth_seg_8u.shape[:2]),dtype='uint8')
        areas=[cv2.contourArea(c) for c in contours] 
        indice=np.argsort(areas)
                
        contour_pos1=np.mean(contours[indice[-1]],0)[0] #the first largest
        contour_pos2=np.mean(contours[indice[-2]],0)[0] #the second largest
        if contour_pos1[1]<contour_pos2[1]:
            c=contours[indice[-1]]
            cv2.drawContours(contour_mask,[c],-1,255,-1)
        else:
            c=contours[indice[-2]]
            cv2.drawContours(contour_mask,[c],-1,255,-1)
                
        hand_seg_d=cv2.bitwise_and(depth_seg_8u,depth_seg_8u,mask=contour_mask)
        depth_seg[hand_seg_d==0]=0
                
        return depth_seg.copy()
         
    
    def segmentIR(self,window):
        # (b-1) delete from d_maximum and blue band
        ir_seg=np.uint8(np.copy(self.ir)) #check if  maximum is bigger 255
        ir_seg=cv2.inRange(ir_seg,self.ir_range[0],self.ir_range[1])              
        ir_seg[self.h_mask>0]=0 #check!
        ir_seg[self.depth>self.d_maximum]=0
        
        #return ir_seg
                
        window_mask=np.zeros(ir_seg.shape[:2],'uint8')
        window_mask[window[2]:window[3],window[0]:window[1]]=1
        ir_seg[window_mask==0]=0
                
        # (b-2)extract contour
        ir_seg_8u=np.uint8(ir_seg)
        ir_seg_8u=cv2.dilate(ir_seg_8u,np.ones((3,3),np.uint8),iterations=1)
        contours, hierachy = cv2.findContours(ir_seg_8u, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                
        # (b-3) select a specific contour
        contour_mask=np.zeros((ir_seg_8u.shape[:2]),dtype='uint8')
        areas=[cv2.contourArea(c) for c in contours]                
        indice=np.argsort(areas)
        c=contours[indice[-1]]
        cv2.drawContours(contour_mask,[c],-1,255,-1)
                
        hand_seg_ir=cv2.bitwise_and(ir_seg_8u,ir_seg_8u,mask=contour_mask)
                
        ir_seg=self.ir.copy()
        ir_seg[hand_seg_ir==0]=0
          
        return ir_seg.copy()
        
        


