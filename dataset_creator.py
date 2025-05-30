nimport math
import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize mediapipe pose class and drawing utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.1, model_complexity=1)

# Calculate angle between 3 points function
def calculateAngle(p1, p2, p3):
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3

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

def detect_pose(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        h, w, _ = image.shape
        landmarks = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in results.pose_landmarks.landmark]
        return landmarks
    return None

# Folder containing images
path = '/Users/simaypay/Desktop/Yoga_Classifier-main/TRAIN'

columns = ["Label","left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle",
           "left_knee_angle","right_knee_angle","angle_for_ardhaChandrasana1","angle_for_ardhaChandrasana2",
           "hand_angle","left_hip_angle","right_hip_angle","neck_angle_uk","left_wrist_angle_bk","right_wrist_angle_bk"]

table = []

for foldername in os.listdir(path):
    folder_path = os.path.join(path, foldername)
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder_path}")
        for filename in os.listdir(folder_path):
            print(filename)
            
            end = os.path.splitext(filename.lower())[1]
            if end in ['.jpg', '.jpeg', '.png']:
        # process the image

                img_path = os.path.join(folder_path, filename)
                print(f"Processing image: {img_path}")
                image = cv2.imread(img_path)
                
                landmarks = detect_pose(image, pose)
                if landmarks is None:
                    print(f"No landmarks detected in: {img_path}")
                    continue
                angle_list = angles_finder(landmarks)
                row = [foldername] + angle_list
                table.append(row)

pose.close()

if table:
    dataframe = pd.DataFrame(table, columns=columns)
    dataframe.to_csv('dataset_images.csv', index=False)
    print("Data saved to dataset_images.csv")
    print(dataframe.head())
else:
    print("No pose landmarks detected in any image. CSV will be empty.")
