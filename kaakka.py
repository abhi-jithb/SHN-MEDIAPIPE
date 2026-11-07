"""
Kaakka Shoo! - Step 1: See yourself as a scarecrow
Goal: get MediaPipe Pose running and understand landmarks.
"""
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

# Landmark IDs we care about (MediaPipe Pose has 33 points, 0-32)
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    print("Camera on! Stand back so your arms are visible. Press 'q' to quit.")
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)               # mirror view, feels natural
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV is BGR, MediaPipe wants RGB
        results = pose.process(rgb)

        if results.pose_landmarks:
            # 1. Draw the full skeleton
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                   mp_pose.POSE_CONNECTIONS)

            lm = results.pose_landmarks.landmark

            # 2. Highlight the wrists and print their y value
            for idx in (L_WRIST, R_WRIST):
                # landmarks are normalized 0.0-1.0 -> multiply to get pixels
                x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                cv2.circle(frame, (x, y), 15, (0, 255, 255), -1)
                cv2.putText(frame, f"y={lm[idx].y:.2f}", (x + 20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 3. Our first "rule": are both wrists above the shoulders?
            #    Remember: y = 0 is the TOP of the image, so "above" means smaller y
            arms_up = (lm[L_WRIST].y < lm[L_SHOULDER].y and
                       lm[R_WRIST].y < lm[R_SHOULDER].y)
            if arms_up:
                cv2.putText(frame, "ARMS UP! SHOO!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 4)

        cv2.imshow("Kaakka Shoo!", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()