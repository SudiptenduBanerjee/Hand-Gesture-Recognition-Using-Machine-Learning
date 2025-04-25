import os
import pickle
import mediapipe as mp
import cv2
# import matplotlib.pyplot as plt # Not used, can be removed

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- CHANGE HERE ---
# Limit detection to only one hand
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
# --- END CHANGE ---

DATA_DIR = './data'

data = []
labels = []
print("Starting dataset creation...") # Add some feedback
processed_images = 0
skipped_images = 0

for dir_ in os.listdir(DATA_DIR):
    print(f"Processing directory: {dir_}")
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path): # Skip files, only process directories
         continue
    for img_path in os.listdir(dir_path):
        full_img_path = os.path.join(dir_path, img_path)
        data_aux = []
        x_ = []
        y_ = []

        try:
            img = cv2.imread(full_img_path)
            if img is None:
                print(f"Warning: Could not read image {full_img_path}. Skipping.")
                skipped_images += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Now results.multi_hand_landmarks will have at most 1 element
            if results.multi_hand_landmarks:
                # Since max_num_hands=1, we can directly access the first hand
                hand_landmarks = results.multi_hand_landmarks[0]

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Check if landmarks were actually found before calculating min
                if not x_ or not y_:
                     print(f"Warning: No landmarks appended for {full_img_path} despite detection. Skipping.")
                     skipped_images += 1
                     continue

                min_x = min(x_)
                min_y = min(y_)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    # Normalize relative to the hand's bounding box
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)

                # Ensure data_aux is not empty before appending
                if data_aux:
                     data.append(data_aux)
                     labels.append(dir_)
                     processed_images += 1
                else:
                     print(f"Warning: Empty data_aux for {full_img_path}. Skipping.")
                     skipped_images += 1

            else: # No hands detected in this image
                # print(f"Info: No hands detected in {full_img_path}. Skipping.") # Optional: print info
                skipped_images += 1

        except Exception as e:
            print(f"Error processing image {full_img_path}: {e}")
            skipped_images += 1


print(f"Dataset creation finished. Processed images: {processed_images}, Skipped images: {skipped_images}")
if not data:
    print("Error: No data was collected. Check your data directory and images.")
else:
    print(f"Saving data for {len(data)} samples.")
    # Save the data
    f = open('data.pickle', 'wb')
    pickle.dump({'data': data, 'labels': labels}, f)
    f.close()
    print("data.pickle saved successfully.")