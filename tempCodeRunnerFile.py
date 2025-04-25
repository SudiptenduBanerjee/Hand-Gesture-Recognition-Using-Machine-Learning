import pickle
import cv2
import mediapipe as mp
import numpy as np
import time # For potential delays if needed

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# --- Try different camera indices ---
cap = cv2.VideoCapture(0) # START WITH 0
if not cap.isOpened():
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Trying camera index 2...")
        cap = cv2.VideoCapture(2) # Try 2 as a last resort
        if not cap.isOpened():
            print("ERROR: Could not open ANY camera. Check connection/index/permissions.")
            exit() # Stop if no camera works

print(f"Successfully opened camera index: {cap.get(cv2.CAP_PROP_POS_FRAMES)}") # Check which index worked


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Use settings appropriate for video and matching your training data ---
hands = mp_hands.Hands(static_image_mode=False, # False for video stream processing
                       max_num_hands=1,        # IMPORTANT: Match your training data (use 1 if trained on 1 hand)
                       min_detection_confidence=0.5, # Slightly higher confidence might be better for real-time
                       min_tracking_confidence=0.5)

# --- Check how your model predicts ---
# If model predicts numbers (0, 1, 2), use the dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C'}
predicts_numbers = True # Set to False if model predicts 'A', 'B', 'L' directly

# If model predicts strings ('A', 'B', 'L'), comment out labels_dict and set predicts_numbers = False
# predicts_numbers = False

print("Starting inference... Press 'q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    # --- ESSENTIAL FIX: Check if frame was read successfully ---
    if not ret:
        print("Warning: Failed to capture frame. Check camera connection/permissions.")
        # You might want to wait briefly or break if this happens repeatedly
        time.sleep(0.5)
        # break # Uncomment this to stop if the camera fails permanently
        continue # Skip the rest of this loop iteration and try again
    # --- END ESSENTIAL FIX ---

    # Frame is valid, proceed with processing
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Improve performance by making image non-writeable before processing
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True # Allow drawing again

    # Convert back to BGR for drawing with OpenCV
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


    if results.multi_hand_landmarks:
        # Since max_num_hands=1, we process only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Extract landmarks and prepare data
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        if not x_ or not y_:
            # Should not happen if landmarks are detected, but good to check
            print("Warning: Landmarks detected but coordinate lists empty.")
            continue

        min_x = min(x_)
        min_y = min(y_)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min_x) # Relative coordinates
            data_aux.append(y - min_y)

        # Ensure the feature vector has the correct length (e.g., 21 landmarks * 2 coords = 42)
        EXPECTED_FEATURES = 42 # Adjust if your feature extraction is different
        if len(data_aux) == EXPECTED_FEATURES:
             # Calculate bounding box with padding, clamping to image boundaries
             x1 = max(0, int(min_x * W) - 10)
             y1 = max(0, int(min_y * H) - 10)
             x2 = min(W - 1, int(max(x_) * W) + 10) # Ensure x2 < W
             y2 = min(H - 1, int(max(y_) * H) + 10) # Ensure y2 < H


             # Make prediction
             prediction = model.predict([np.asarray(data_aux)])

             # Get character based on prediction type
             if predicts_numbers:
                 predicted_character = labels_dict.get(int(prediction[0]), '?') # Use .get for safety
             else:
                 predicted_character = str(prediction[0]) # Directly use the predicted string


             # Draw bounding box and text
             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box
             cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA) # Green text
        else:
            print(f"Warning: Feature count mismatch. Expected {EXPECTED_FEATURES}, got {len(data_aux)}. Skipping prediction.")


    # Display the frame regardless of detection
    cv2.imshow('frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit key pressed.")
        break

# Release resources
print("Releasing camera and closing windows...")
cap.release()
cv2.destroyAllWindows()
