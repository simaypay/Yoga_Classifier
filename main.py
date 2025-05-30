import cv2
import mediapipe as mp
import joblib
import math 

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
    angle_for_ardhaChandrasana1 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    angle_for_ardhaChandrasana2 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    hand_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
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
    return [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle, angle_for_ardhaChandrasana1, angle_for_ardhaChandrasana2,
            hand_angle, left_hip_angle, right_hip_angle, neck_angle_uk, left_wrist_angle_bk, right_wrist_angle_bk]
"""
def detect_pose(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        h, w, d = image.shape
        landmarks = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in results.pose_landmarks.landmark]
        print(landmarks)
        return landmarks
    return None"""
 #-------------------------------- feedback calculation ---------------------------------------#
ideal_angles_HalfMoon = [171.59, 189.31, 98.08, 94.66, 179.63, 183.02, 253.06, 106.74, 175.30, 121.65, 238.57, 309.07, 83.05, 280.71]
ideal_angles_Butterfly = [194.87, 165.46, 32.57, 32.31, 343.11, 17.54, 304.72, 52.07, 49.08, 74.14, 283.82, 305.93, 191.05, 115.79]
ideal_angles_Downward_Dog = [168.99, 166.94, 174.86, 187.43, 178.0, 178.97, 310.85, 49.03, 111.66, 81.51, 80.85, 206.01, 80.67, 79.9]
ideal_angles_Dancer = [173.19, 182.48, 127.04, 109.22, 146.81, 237.03, 202.31, 156.94, 203.75, 128.35, 238.7, 281.3, 98.15, 298.18]
ideal_angles_Triangle = [169.49, 187.7, 87.46, 114.49, 178.95, 176.47, 283.95, 85.0, 182.42, 54.19, 146.7, 326.85, 94.41, 174.23]
ideal_angles_Goddess = [166.89, 192.5, 91.66, 86.3, 242.31, 116.53, 261.62, 98.53, 151.96, 111.47, 249.03, 295.76, 131.99, 229.02]
ideal_angles_Warrior = [178.93, 179.94, 104.23, 104.73, 209.99, 143.94, 254.51, 105.94, 182.47, 115.43, 246.03, 299.39, 93.24, 264.85]
ideal_angles_Tree = [210.82, 146.19, 124.92, 119.88, 247.84, 128.14, 242.33, 162.3, 197.31, 150.54, 202.66, 289.07, 192.77, 169.7]

thresholds_good = [10, 10, 15, 15, 10, 10, 20, 20, 15, 15, 15, 10, 20, 20] 
thresholds_warn = [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 25, 20, 30, 30]
''' ADAPT VALUES '''

user_angles = angles_finder(landmarks) 


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
# ----------------------- same as dataset creator functions -------------

mp_pose= mp.solutions.pose
skeleton0= mp.solutions.drawing_utils

pose=mp_pose.Pose()

cam= cv2.VideoCapture(0)
model= joblib.load("mymodel.pkl")
while cam.isOpened():
    success , frame = cam.read()

    
    img_rgb= cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    img_copy= frame.copy()
    landmarks= pose.process(img_rgb)
    (h,w,d)= img_copy.shape
    if landmarks.pose_landmarks:

        skeleton0.draw_landmarks(img_copy, landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        coords=landmarks.pose_landmarks.landmark
        
        standard_coords = [((w*els.x) , (h*els.y), (d* els.z)) for els in coords]
        
        angles= angles_finder(standard_coords)
        
        prediction=model.predict([angles])
        print(prediction[0])
    
    cv2.imshow('Camera', img_copy)


    
    if cv2.waitKey(1) == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()

