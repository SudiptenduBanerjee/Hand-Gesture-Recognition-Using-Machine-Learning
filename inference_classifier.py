import pickle
import cv2
import mediapipe as mp
import numpy as np
import time # Import time for potential debugging/pausing

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# --- Try different camera indices ---
cap = cv2.VideoCapture(0) # Try 0 first
if not cap.isOpened():
    print("Trying camera index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Trying camera index 2...")
        cap = cv2.VideoCapture(2)
        if not cap.isOpened():
            print("ERROR: Could not open any camera. Check connection/index.")
            exit() # Exit if no camera works


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Use settings appropriate for video and matching your training data ---
hands = mp_hands.Hands(static_image_mode=False, # False for video
                       max_num_hands=1,        # Match training if it used 1 hand
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.5) # Added tracking confidence

# Ensure labels_dict matches the output of your training script
# (e.g., if your model predicts numbers 0, 1, 2)
# labels_dict = {0: 'A', 1: 'B', 2: 'L'}
# OR if your model predicts the letters directly (less common with sklearn directly)
# you might need to adjust how you get the predicted_character
# Check how y_train/y_test looked in train_classifier.py. If they were 'A', 'B', 'L',
# the model likely predicts those directly. If they were numbers, use the dict.

# Let's assume your model predicts strings 'A', 'B', 'L' based on the training script
# If it predicts numbers 0, 1, 2 corresponding to A, B, L, uncomment the dict above
# and use: predicted_character = labels_dict[int(prediction[0])]

print("Starting inference... Press 'q' to quit.")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    # --- FIX: Check if frame was read successfully ---
    if not ret:
        print("Warning: Failed to capture frame. Skipping.")
        # Optionally add a small delay or break completely
        # time.sleep(0.1)
        # break # Use break if the camera seems permanently disconnected
        continue # Try again on the next loop iteration
    # --- END FIX ---

    # Now it's safe to use frame
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Since max_num_hands=1, we process only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw landmarks
        mp_drawing.draw_landmarks(
            frame,  # image to draw
            hand_landmarks,  # model output
            mp_hands.HAND_CONNECTIONS,  # hand connections
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Extract landmarks and prepare data in the same loop
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)

        # Check if landmarks were actually extracted before proceeding
        if not x_ or not y_:
            print("Warning: Landmarks detected but coordinate lists are empty.")
            cv2.imshow('frame', frame) # Show frame even if processing fails
            if cv2.waitKey(1) & 0xFF == ord('q'):
                 break
            continue

        min_x = min(x_)
        min_y = min(y_)

        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min_x)
            data_aux.append(y - min_y)

        # Check if data_aux has the expected number of features (e.g., 42)
        # This guards against unexpected issues in landmark extraction
        if len(data_aux) == 42: # Adjust 42 if your feature count is different
            # Calculate bounding box coordinates (optional padding -10)
            # Ensure coordinates stay within frame boundaries
            x1 = max(0, int(min_x * W) - 10)
            y1 = max(0, int(min_y * H) - 10)
            x2 = min(W, int(max(x_) * W) + 10) # Use +10 for max extent
            y2 = min(H, int(max(y_) * H) + 10) # Use +10 for max extent

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = str(prediction[0]) # Get the predicted label directly

            # Draw bounding box and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box, thickness 2
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA) # Red text

        else:
             print(f"Warning: Incorrect number of features extracted ({len(data_aux)}). Skipping prediction.")


    # Display the frame
    cv2.imshow('frame', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
print("Exiting...")
cap.release()
cv2.destroyAllWindows()