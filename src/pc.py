from kivy.uix.image import Image
from kivymd.app import MDApp
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture 
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.utils import platform
import cv2
import os
import numpy as np
from datetime import datetime
class BarbellBuddy(MDApp):
    def build(self):
        #Build app
        self.image = Image()
        layout = MDBoxLayout(orientation = 'vertical')
        layout.add_widget(self.image)
        
        import tflite_runtime.interpreter as tf
        self.interpreter = tf.Interpreter(model_path='model.tflite')
        self.interpreter.allocate_tensors()
        self.cap = cv2.VideoCapture(0)

    
        #Initialize pose variables
        self.t = datetime.now()
        self.leftleg = False
        self.rightleg = False   

        #Loop update()
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    def update(self,dt):
        
        #Read in frame
        _,frame = self.cap.read()
        if not self.cap.isOpened():
            print("Error: Could not open camera")
        else:
            #frame1 = frame.copy()
            frame1_resized = cv2.resize(frame, (192, 192))

            #Calculate padding
            height, width = frame1_resized.shape[:2]
            top_pad = (192 - height) // 2
            bottom_pad = 192 - height - top_pad
            left_pad = (192 - width) // 2
            right_pad = 192 - width - left_pad

            #Apply padding to the resized image
            frame1_resized_padded = cv2.copyMakeBorder(frame1_resized, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=0)

            #Convert data type to float32
            input_data = frame1_resized_padded.astype(np.float32)
            input_data = np.expand_dims(input_data, axis=0)
            dims = frame.shape

            #Pre Initilize Landmarks
            input_details = self.interpreter.get_input_details()
            output_details= self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]['index'],input_data)
            self.interpreter.invoke()
            keypoints = self.interpreter.get_tensor(output_details[0]['index'])

            #Hip and knee coords
            left_knee = keypoints[0][0][13]
            lkc = left_knee[2]
            left_knee = left_knee[:2]*[dims[1],dims[2]]

            right_knee = keypoints[0][0][14]
            rkc = right_knee[2]
            right_knee = right_knee[:2]*[dims[1],dims[2]]

            left_hip = keypoints[0][0][11]
            lhc = left_hip[2]
            left_hip = left_hip[:2]*[dims[1],dims[2]]

            right_hip = keypoints[0][0][12]
            rhc = right_hip[2]
            right_hip = right_hip[:2]*[dims[1],dims[2]]

            #Draw skeleton
            self.draw_keypoints(frame,keypoints,.2)
            self.draw_edges(frame,keypoints,.2)
            #Detect depth
            leftdetect = (left_hip[0] > left_knee[0]) and (lhc>.2 and lkc>.2)
            rightdetect = (right_hip[0] > right_knee[0])and (rhc>.2 and rkc>.2)
            if leftdetect or rightdetect:
                self.t = datetime.now()
                if leftdetect:
                    self.leftleg = True
                if rightdetect:
                    self.rightleg = True     
            if (datetime.now()-self.t).total_seconds() < 5:
                self.lights(frame,self.leftleg,self.rightleg)
            if (datetime.now()-self.t).total_seconds() > 5:
                self.leftleg = False
                self.rightleg = False
            if frame is not None:
                buffer = cv2.flip(frame,0).tostring()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
                texture.blit_buffer(buffer, colorfmt = 'bgr', bufferfmt='ubyte')
                self.image.texture = texture
        
    #Draw keypoints on image
    def draw_keypoints(self,frame,keypoints,conf):
        y,x,z = frame.shape
        shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf>conf:
                cv2.circle(frame, (int(kx),int(ky)), 4,(0,0,255),-1)
    #Draw edges for keypoints
    def draw_edges(self,frame,keypoints,conf):
        EDGES = {
            (0, 1): 'm',(0, 2): 'c',
            (1, 3): 'm',(2, 4): 'c',
            (0, 5): 'm',(0, 6): 'c',
            (5, 7): 'm',(7, 9): 'm',
            (6, 8): 'c',(8, 10): 'c',
            (5, 6): 'y',(5, 11): 'm',
            (6, 12): 'c',(11, 12): 'y',
            (11, 13): 'm',(13, 15): 'm',
            (12, 14): 'c',(14, 16): 'c'
        }
        y,x,z = frame.shape
        shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
        for edge, color in EDGES.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            if (c1 > conf) & (c2 > conf):
                cv2.line(frame, (int(x1),int(y1)), (int(x2),int(y2)),(255,0,0),2)
    #Draw lights on image
    def lights(self,frame,left,right):
        if left == True and right == False:
            cv2.circle(frame, (100,100), 30, (255,255,255),-1)
            cv2.circle(frame, (200,100), 30, (255,255,255),-1)
            cv2.circle(frame, (300,100), 30, (0,0,255),-1)
        elif left == False and right == True:
            cv2.circle(frame, (100,100), 30, (0,0,255),-1)
            cv2.circle(frame, (200,100), 30, (255,255,255),-1)
            cv2.circle(frame, (300,100), 30, (255,255,255),-1)
        elif left == True and right== True:
            cv2.circle(frame, (100,100), 30, (255,255,255),-1)
            cv2.circle(frame, (200,100), 30, (255,255,255),-1)
            cv2.circle(frame, (300,100), 30, (255,255,255),-1)        
if __name__ == '__main__':
    BarbellBuddy().run()