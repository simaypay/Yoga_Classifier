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
 #--------------------------------same as dataset creator functions---------------------------------------#


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

