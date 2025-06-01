import cv2
import numpy as np
import mediapipe as mp
import joblib
import math 
import pandas as pd
import os
import warnings
warnings.filterwarnings ("ignore", category=UserWarning, module="sklearn")

def calculateAngle(p1, p2, p3):
    (x1, y1, z1 )= p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def angles_finder(landmarks):
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    angle_for_half_moon1 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    angle_for_half_moon2 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    neck_angle_uk = calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    left_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    angles_list = [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle, angle_for_half_moon1, angle_for_half_moon2,
             left_hip_angle, right_hip_angle, neck_angle_uk, left_wrist_angle_bk, right_wrist_angle_bk]

    columns = ["left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle",
           "left_knee_angle","right_knee_angle","angle_for_half_moon1","angle_for_half_moon2",
           "left_hip_angle","right_hip_angle","neck_angle_uk","left_wrist_angle_bk","right_wrist_angle_bk"]

    angles_dataframe = pd.DataFrame([angles_list], columns=columns)
    return angles_dataframe , angles_list
 #--------------------------------same as dataset creator functions---------------------------------------#


ideal_angles = {"Half_Moon_left": [171.59, 189.31, 98.08, 94.66, 179.63, 183.02, 253.06, 106.74,  121.65, 238.57, 309.07, 83.05, 280.71],
                "Half_Moon_right":[45.0, 120.0, 90.0, 75.0, 135.0, 160.0, 110.0, 95.0, 85.0, 70.0, 100.0, 140.0, 130.0],

                "Butterfly": [194.87, 165.46, 32.57, 32.31, 343.11, 17.54, 304.72, 52.07,  74.14, 283.82, 305.93, 191.05, 115.79],
                "Downward_dog": [168.99, 166.94, 174.86, 187.43, 178.0, 178.97, 310.85, 49.03,  81.51, 80.85, 206.01, 80.67, 79.9],
                "Dancer_right": [173.19, 182.48, 127.04, 109.22, 146.81, 237.03, 202.31, 156.94,  128.35, 238.7, 281.3, 98.15, 298.18],
                "Dancer_left":[45.0, 120.0, 90.0, 75.0, 135.0, 160.0, 110.0, 95.0, 85.0, 70.0, 100.0, 140.0, 130.0],

                "Triangle": [169.49, 187.7, 87.46, 114.49, 178.95, 176.47, 283.95, 85.0,  54.19, 146.7, 326.85, 94.41, 174.23],
                "Goddess": [166.89, 192.5, 91.66, 86.3, 242.31, 116.53, 261.62, 98.53,  111.47, 249.03, 295.76, 131.99, 229.02],
                "Warrior_left": [178.93, 179.94, 104.23, 104.73, 209.99, 143.94, 254.51, 105.94,  115.43, 246.03, 299.39, 93.24, 264.85],
                "Warrior_right":[45.0, 120.0, 90.0, 75.0, 135.0, 160.0, 110.0, 95.0, 85.0, 70.0, 100.0, 140.0, 130.0],

                "Tree_right": [210.82, 146.19, 124.92, 119.88, 247.84, 128.14, 242.33, 162.3,  150.54, 202.66, 289.07, 192.77, 169.7],
                "Tree_left":[45.0, 120.0, 90.0, 75.0, 135.0, 160.0, 110.0, 95.0, 85.0, 70.0, 100.0, 140.0, 130.0]}

thresholds_good = [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20]   
thresholds_warn = [20, 20, 25, 25, 20, 20, 30, 30,  25, 25, 20, 30, 30]

def compare_angles(user_angles, ideal_angles, threshold_good, threshold_warn):
    feedback_list = []

    length = len(user_angles)
    for i in range(length):
        error = abs(user_angles[i] - ideal_angles[i])
       
        if error <= threshold_good[i]:
            feedback = "✅"
        elif error <= threshold_warn[i]:
            feedback = "⚠️"
        else:
            feedback = "❌"

        feedback_list.append(feedback)
    return feedback_list


def load_emoji(filename, size=(24, 24)):
    path = os.path.join("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images", filename)
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)  
    if emoji is None:
        raise FileNotFoundError(f"Could not load {filename}")
    return cv2.resize(emoji, size)



def overlay_emoji(frame, emoji_img, x, y, size=24):
    frame_h, frame_w = frame.shape[:2]

    # Resize emoji
    emoji_img = cv2.resize(emoji_img, (size, size), interpolation=cv2.INTER_AREA)
    emoji_h, emoji_w = emoji_img.shape[:2]

    # Adjust x, y if too close to edges
    if x + emoji_w > frame_w:
        x = frame_w - emoji_w
    if y + emoji_h > frame_h:
        y = frame_h - emoji_h
    if x < 0 or y < 0:
        return  # Emoji won't fit

    bgr = emoji_img[:, :, :3]
    
    
        
    alpha = emoji_img[:, :, 3] / 255.0
    
        

    roi = frame[y:y+emoji_h, x:x+emoji_w]

    for c in range(3):
        roi[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c])
    
"""

def overlay_emoji(frame, emoji_img, x, y, size=24):
    x, y = int(x), int(y)
    emoji_img = cv2.resize(emoji_img, (size, size))
    h, w = emoji_img.shape[:2]

    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return  # Don't draw if out of bounds

    roi = frame[y:y+h, x:x+w]

    if emoji_img.shape[2] == 4:
        # Handle transparent emoji
        alpha = emoji_img[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (alpha * emoji_img[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8)
    else:
        # Handle non-transparent emoji (fallback)
        roi[:] = emoji_img[:, :, :3]

    frame[y:y+h, x:x+w] = roi

"""

check_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/check.png", size=(24,24)) # Load all 3 emojis
warn_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/cross.png", size=(24,24))
cross_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/warning.png", size=(24,24))

#-------------------------------feedback system----------------------------------#

mp_pose= mp.solutions.pose
mp_skeleton= mp.solutions.drawing_utils

pose=mp_pose.Pose()

model= joblib.load("mymodel.pkl")

cam= cv2.VideoCapture(0)


joints = [ #these are indexes
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE, 
                mp_pose.PoseLandmark.LEFT_ANKLE,   
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
#just variables
frame_no=0
last_prediction = None
last_feedback_list = None


while cam.isOpened():
    success , frame = cam.read()

    
    img_rgb= cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    img_copy= frame.copy()
    (h,w,d)= img_copy.shape
    
    landmarks= pose.process(img_rgb)
    
    
    if landmarks.pose_landmarks:
        mp_skeleton.draw_landmarks(img_copy, landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        frame_no+=1

        if frame_no %20==0 :

            coords=landmarks.pose_landmarks.landmark  #the coordinates as a list but normalized (0,1)
        
            standard_coords = [((w*els.x) , (h*els.y), (d* els.z)) for els in coords] #only x,y,z and real coordinates
        
            angles_dataframe ,angles= angles_finder(standard_coords) #angles is dataframe to predict 

        #the model prediction
            prediction=model.predict(angles_dataframe)
            

        


        #feedback down here
            ideal_list = ideal_angles[prediction[0]]
            feedback_list = compare_angles(angles, ideal_list, thresholds_good, thresholds_warn)


        #cacheing for frame reduction
            last_prediction= prediction
            last_feedback_list= feedback_list
            
        if last_prediction is not None and last_feedback_list is not None:
            for i, joint in enumerate(joints):
                landmark = coords[joint.value]     #coords = landmarks.pose_landmarks.landmark (list)
                if landmark.visibility < 0.5:
                    continue
                x = int(coords[joint.value].x * w)
                y = int(coords[joint.value].y * h)
    
                feedback = last_feedback_list[i]
                
                if feedback == "✅":
                    overlay_emoji(img_copy, check_img, x, y)
                elif feedback == "⚠️":
                    overlay_emoji(img_copy, warn_img, x, y)
                elif feedback == "❌":
                    overlay_emoji(img_copy, cross_img, x, y)
                
            overall_score = (last_feedback_list.count("✅") + (last_feedback_list.count("⚠️"))/3) / len(last_feedback_list) * 100
    
            cv2.putText(img_copy, f"Pose: {last_prediction[0]}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(img_copy, f"Score: {overall_score:.1f}%", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Yoga Pose Feedback", img_copy)
    
    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
pose.close()
cv2.destroyAllWindows()

