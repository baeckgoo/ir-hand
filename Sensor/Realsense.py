## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

class Realsense():
    def __init__(self):
        # Create a pipeline
        self.pipeline = rs.pipeline()

        #Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)
        
        # Start streaming
        profile = self.pipeline.start(config)
    
        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)

        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        clipping_distance_in_meters = 1 #1 meter
        self.clipping_distance = clipping_distance_in_meters / self.depth_scale
        
        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        
        #align_to = rs.stream.color
        align_to=rs.stream.depth
        #align_to=rs.stream.infrared 
        
        self.align = rs.align(align_to)

    def run(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
            
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        ir_frame= frames.get_infrared_frame()    
        
        #depth
        depth_image = np.asarray(aligned_depth_frame.get_data(),dtype=np.float32)
        self.depth_image=depth_image * self.depth_scale * 1000
        #ir
        self.ir_image= np.asarray(ir_frame.get_data(), dtype=np.uint16)
        
        #color
        self.color_image = np.asanyarray(color_frame.get_data())

    def getIrImage(self):
        return self.ir_image     

    def getDepthImage(self):
        return self.depth_image

    def getColorImage(self):
        return self.color_image   

    def release(self):
        self.pipeline.stop()
        
        
if __name__=="__main__":
    realsense=Realsense()
    
    try:
        while True:
            realsense.run()
            
            depth=realsense.getDepthImage()
            ir=realsense.getIrImage()
            color=realsense.getColorImage()
            cv2.imshow('depth',np.uint8(depth))
            cv2.imshow('ir',np.uint8(ir))
            cv2.imshow('color',color)
            
            dst=cv2.addWeighted(np.uint8(depth),0.5,np.uint8(ir),1.0,0)
            cv2.imshow('dst',dst)
            
            
            
            cv2.waitKey(1)
            
    finally:
        realsense.release()
        
    
    
    
    