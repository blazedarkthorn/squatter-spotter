#Pose Estimator reference: https://www.hackersrealm.net/post/realtime-human-pose-estimation-using-python

#Kivy Utilities
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivy.clock import Clock
from kivy.graphics.texture import Texture 
from kivymd.uix.boxlayout import MDBoxLayout
import cv2
import tensorflow as tf
import numpy as np

class BarbellBuddy(MDApp):
    def build(self):
        self.interpreter = tf.lite.Interpreter(model_path= 'lite-model_movenet_singlepose_lightning_3.tflite')
        self.interpreter.allocate_tensors()
        #Build App
        self.image = Image()
        layout = MDBoxLayout(orientation = 'vertical')
        layout.add_widget(self.image)

        #Initialize Camera and Pose Variables
        self.cap = cv2.VideoCapture(0)
        self.depth = False
        #Loop Update()
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    def update(self,dt):
        
        #Pose Estimator        
        _,frame = self.cap.read()
        frame1 = frame.copy()
        frame1 = tf.image.resize_with_pad(np.expand_dims(frame1,axis=0),192,192)
        input_image = tf.cast(frame1,dtype=tf.float32)
        dims = frame.shape

        #Pre Initilize Landmarks
        input_details = self.interpreter.get_input_details()
        output_details= self.interpreter.get_output_details()
        self.interpreter.set_tensor(input_details[0]['index'],np.array(input_image))
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(output_details[0]['index'])

        #Hip and Knee Coords
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

        #Draw Skeleton
        self.draw_keypoints(frame,keypoints,.2)
        self.draw_edges(frame,keypoints,.2)
        #Detect Depth

        if (((left_hip[0] > left_knee[0]) and (lhc>.2 and lkc>.2)) or (right_hip[0] > right_knee[0]and (rhc>.2 and rkc>.2))):
            text = "Depth"
            if self.depth == False:
                self.depth = True
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, text, (50, 50), 
                        font, 1, 
                        (0, 255, 255), 
                        2, 
                        cv2.LINE_4)
        else:
            self.depth = False
        buffer = cv2.flip(frame,0).tostring()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt = 'bgr')
        texture.blit_buffer(buffer, colorfmt = 'bgr', bufferfmt='ubyte')
        self.image.texture = texture
        

    def draw_keypoints(self,frame,keypoints,conf):
        y,x,z = frame.shape
        shaped = np.squeeze(np.multiply(keypoints,[y,x,1]))
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf>conf:
                cv2.circle(frame, (int(kx),int(ky)), 4,(0,0,255),-1)
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
if __name__ == '__main__':
    BarbellBuddy().run()