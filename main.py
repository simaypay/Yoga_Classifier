import cv2
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
    return angles_dataframe
 #--------------------------------same as dataset creator functions---------------------------------------#


ideal_angles = {"Half Moon": [180, 180, 95, 100, 180, 190, 260, 90, 180, 180, 280, 290, 200, 200],
                "Butterfly": [194, 165, 27, 27, 343, 16, 304, 52, 55, 80, 280, 305, 191, 115],
                "Downward Dog": [170, 165, 175, 187, 178, 179, 310, 49, 111, 81, 80, 206, 80, 79],
                "Dancer": [163, 176, 190, 65, 180, 270, 202, 160, 210, 120, 240, 230, 130, 300],
                "Triangle": [180, 185, 85, 115, 178, 176, 283, 85, 190, 60, 150, 326, 94, 174],
                "Goddess": [300, 50, 40, 40, 250, 110, 261, 98, 151, 111, 249, 295, 131, 229],
                "Warrior": [178.93, 179.94, 80, 104.73, 240, 180, 260, 10, 170, 140, 260, 330, 93, 264],
                "Tree": [129,231, 172, 165, 180, 70, 350, 20, 280, 172, 240, 289, 180, 170]}

threshold_good = [20, 20, 20, 20, 20, 20, 60, 70, 100, 20, 35, 50, 40, 40]
threshold_warn = [30, 30, 30, 30, 30, 30, 180, 180, 150, 30, 45, 60, 60, 60]
# ADAPT VALUES, or make a dictionary (each entry, a pose)
''' ADAPT VALUES + add to lists '''


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

ef load_emoji(filename, size=(24,24)):
    path = os.path.join("feedback images", filename) 
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        raise FileNotFoundError(f"Could not load {filename}")
    return cv2.resize(emoji, size, interpolation=cv2.INTER_AREA)


check_img = load_emoji("check.png") # Load all 3 emojis
warn_img = load_emoji("warning.png")
cross_img = load_emoji("cross.png")

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

#-------------------------------feedback system----------------------------------#

mp_pose= mp.solutions.pose
mp_skeleton= mp.solutions.drawing_utils

pose=mp_pose.Pose()

model= joblib.load("mymodel.pkl")

cam= cv2.VideoCapture(0)


frame_no=0
while cam.isOpened():
    success , frame = cam.read()
    img_rgb= cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    img_copy= frame.copy()
    (h,w,d)= img_copy.shape
    
    landmarks= pose.process(img_rgb)
    
    
    if landmarks.pose_landmarks:
        
        mp_skeleton.draw_landmarks(img_copy, landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        frame_no+=1
        if frame_no %20 :
            coords=landmarks.pose_landmarks.landmark  #the coordinates as a list but normalized (0,1)
        
            standard_coords = [((w*els.x) , (h*els.y), (d* els.z)) for els in coords] #only x,y,z and real coordinates
        
            angles= angles_finder(standard_coords) #angles is dataframe to predict 

            #the model prediction
            prediction=model.predict(angles)
            #print(prediction[0])
    
        #feedback down here
            ideal_list = ideal_angles[prediction[0]]
            
            feedback_list = compare_angles(angles, ideal_list, threshold_good, threshold_warn)

            joints = [
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE, 
                mp_pose.PoseLandmark.LEFT_ANKLE,  
                mp_pose.PoseLandmark.RIGHT_ELBOW,  
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
        
            for i, joint in enumerate(joints):
                landmark = coords[joint.value]
                if landmark.visibility < 0.5:
                    continue
                x = int(coords[joint.value].x * w)
                y = int(coords[joint.value].y * h)
    
                feedback = feedback_list[i]
                
                if feedback == "✅":
                    overlay_emoji(img_copy, check_img, x, y)
                elif feedback == "⚠️":
                    overlay_emoji(img_copy, warn_img, x, y)
                elif feedback == "❌":
                    overlay_emoji(img_copy, cross_img, x, y)
                
            overall_score = (feedback_list.count("✅") + (feedback_list.count("⚠️"))/3) / len(feedback_list) * 100
    
            cv2.putText(img_copy, f"Pose: {prediction[0]}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(img_copy, f"Score: {overall_score:.1f}%", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)


    
    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
pose.close()
cv2.destroyAllWindows()

