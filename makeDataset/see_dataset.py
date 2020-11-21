import numpy as np
import cv2

from tqdm import tqdm
from tqdm import trange


path='/home/yong/ssd/dataset/preprocessed_HIG/train_blur/'
savepath='/home/yong/hdd/HIG/check_preprocessed/'

progressbar=trange(0,10000,leave=True)  
#for frame in range(10000):
for frame in progressbar:
    img=np.load(path+'%d.npy'%frame)
    
    img=(img+1)*128-1
    
    cv2.imshow('img',np.uint8(img))
    cv2.imwrite(savepath+'%d.png'%frame,img)
    
    cv2.waitKey(1)