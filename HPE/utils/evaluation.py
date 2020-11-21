import numpy as np
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluateEmbeddingError(output,target):
    bsize=output.shape[0]
    dim=output.shape[1]
    error=0
    
    #error=torch.norm(output-target)/(bsize)
    error=((output-target)**2).mean()    
      
    return  error
            
    
if __name__=="__main__":    
    output = torch.rand(5,30)
    target = torch.rand(5,30)

    start_time=time.time()
    
    a=evaluateEmbeddingError(output,target)
    
    print('error',a)
    print(a.shape)
    
    print(time.time()-start_time)
    
    
    
    
    
    
    
    