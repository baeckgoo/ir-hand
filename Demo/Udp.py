# reference:  https://wikidocs.net/13905
# reference: https://itsaessak.tistory.com/125
# reference: https://kldp.org/node/2870

import socket

import numpy as np
import time

class UDP:
    def __init__(self,port):
        self.HOST='192.168.0.3'    #gypark's com   #127.0.0.1 for local host
        #self.HOST='143.248.97.96' #hyungil's com
        self.PORT=port

        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    def send(self,data):
        #data_byte=bytes(data)
        data_byte=data
        self.socket.sendto(data_byte,(self.HOST,self.PORT))
        
    def close(self):
        self.socket.close()
        

if __name__=="__main__":
    
    if not 'udp' in locals():
        udp=UDP(89)
    
    
    data=np.zeros(27,'float32')
    data[3]=0.7878
    while(1):
        #data=np.array([1.1,2.2,3.3,4.4],'float32')
        udp.send(data)
        time.sleep(1)
        
        #print(data[3])
    
