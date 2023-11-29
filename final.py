import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

video_capture = cv2.VideoCapture('vid1.mp4')

threshold_line_x = 0.54
step_count = 0
foot_above_threshold = False

check = True
fps = video_capture.get(cv2.CAP_PROP_FPS)
print(fps)

# Set the desired maximum window height
max_window_height = 800

# Get the original video dimensions
# 3 corresponds to CV_CAP_PROP_FRAME_WIDTH
original_width = int(video_capture.get(3))
# 4 corresponds to CV_CAP_PROP_FRAME_HEIGHT
original_height = int(video_capture.get(4))

# Calculate the corresponding window width to maintain the aspect ratio
window_width = int(max_window_height * (original_width / original_height))

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize the frame to fit the window while maintaining the aspect ratio
    frame = cv2.resize(frame, (window_width, max_window_height))

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_frame)

    thresh = results.pose_landmarks.landmark[32]
    if thresh:
        threshold_line_x = results.pose_landmarks.landmark[32].x
    else:
        threshold_line_x = 0.54

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        left_foot_x = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x if landmarks[
            mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility > 0.5 else None

        # Check if the left foot is above the threshold line and was not above it in the previous frame
        if left_foot_x and left_foot_x > threshold_line_x and not foot_above_threshold:
            step_count += 1
            foot_above_threshold = True
        elif left_foot_x and left_foot_x <= threshold_line_x:
            foot_above_threshold = False

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f'Steps: {int(step_count)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        threshold_line_pixel = int(threshold_line_x * frame.shape[1])
        cv2.line(frame, (threshold_line_pixel, 0),
                 (threshold_line_pixel, frame.shape[0]), (0, 0, 255), 2)

    cv2.imshow('MediaPipe Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
