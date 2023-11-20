#kivy imports
from kivy.uix.image import Image
from kivymd.app import MDApp
from kivy.uix.camera import Camera
from kivy.clock import Clock
from kivy.graphics.texture import Texture 
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.utils import platform

#Mediapipe imports
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

#Extra utils
import numpy as np
from datetime import datetime
import cv2

class BarbellBuddy(MDApp):
    def build(self):
        #Build app
        self.image = Image()
        layout = MDBoxLayout(orientation = 'vertical')
        layout.add_widget(self.image)
        model_path = 'model.task'

        #Create a pose landmarker instance with the video mode:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

        #self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('squattest2.mp4')
        #self.cap = cv2.VideoCapture('ss2.png')
    
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
            # Initialize HOG person detector from OpenCV and detect people
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            people, _ = hog.detectMultiScale(frame, winStride=(8, 8))
            frame_center = (frame.shape[1]/2,frame.shape[0]/2)
            distances = [abs((x + w // 2) - frame_center[0]) for _, _, x, _, w, _ in people]
        
            # Get index of the person closest to the center
            closest_person_idx = distances.index(min(distances))
            x, y, w, h = people[closest_person_idx]

            # Crop and process the frame for pose estimation
            cropped_image = people[y:y + h, x:x + w]
            cv2.normalize(cropped_image, cropped_image, 0, 255, cv2.NORM_MINMAX)
            image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            results = self.pose.process(image_rgb)
            if results.pose_landmarks:
             # Render pose landmarks on the image
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                #Hip and knee coords
                left_knee = results.pose_landmarks.landmark[26]

                right_knee = results.pose_landmarks.landmark[25]

                left_hip = results.pose_landmarks.landmark[24]

                right_hip = results.pose_landmarks.landmark[23]
                
                #Detect depth
                leftdetect = (left_hip.y > left_knee.y)
                rightdetect = (right_hip.y > right_knee.y)
                if leftdetect or rightdetect:
                    self.t = datetime.now()
                    if leftdetect:
                        self.leftleg = True
                    if rightdetect:
                        self.rightleg = True
            cv2.rectangle(frame, (int(frame.shape[0]*.1),int(frame.shape[1]*.1)), (int(frame.shape[0]*.2),int(frame.shape[1]*.1)), (0,0,0), int(frame.shape[0]*.075)) 

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

    #Draw lights on image
    def lights(self,frame,left,right):
        if left == True and right == False:
            cv2.circle(frame, ((int(frame.shape[0]*.1),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.15),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.2),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (0,0,255),-1)
        elif left == False and right == True:
            cv2.circle(frame, ((int(frame.shape[0]*.1),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (0,0,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.15),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.2),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
        elif left == True and right== True:
            cv2.circle(frame, ((int(frame.shape[0]*.1),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.15),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)
            cv2.circle(frame, ((int(frame.shape[0]*.2),int(frame.shape[1]*.1))), int(frame.shape[0]*.02), (255,255,255),-1)        
if __name__ == '__main__':
    BarbellBuddy().run()