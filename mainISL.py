import cv2
import mediapipe as mp
import numpy as np
import keras
import time
import os
import pyautogui

# 1. SETUP & SUPPRESS LOGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load your model
model = keras.models.load_model('model.h5')

# 36 Labels (0-9, A-Z) - Ensure this matches your training folder order!
labels = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

# 2. ACTION MAP (Dictionary of commands)
actions = {
    # Edit & File
    'C': {'command': lambda: pyautogui.hotkey('ctrl', 'c'), 'label': 'Copying...'},
    'V': {'command': lambda: pyautogui.hotkey('ctrl', 'v'), 'label': 'Pasting...'},
    'X': {'command': lambda: pyautogui.hotkey('ctrl', 'x'), 'label': 'Cutting...'},
    'Z': {'command': lambda: pyautogui.hotkey('ctrl', 'z'), 'label': 'Undo Action'},
    'Y': {'command': lambda: pyautogui.hotkey('ctrl', 'y'), 'label': 'Redo Action'},
    'A': {'command': lambda: pyautogui.hotkey('ctrl', 'a'), 'label': 'Select All'},
    'S': {'command': lambda: pyautogui.hotkey('ctrl', 's'), 'label': 'Saving...'},
    'P': {'command': lambda: pyautogui.hotkey('ctrl', 'p'), 'label': 'Opening Print'},
    'F': {'command': lambda: pyautogui.hotkey('ctrl', 'f'), 'label': 'Search/Find'},
    
    # Windows & System
    'L': {'command': lambda: pyautogui.hotkey('win', 'l'), 'label': 'Locking PC'},
    'D': {'command': lambda: pyautogui.hotkey('win', 'd'), 'label': 'Showing Desktop'},
    'E': {'command': lambda: pyautogui.hotkey('win', 'e'), 'label': 'File Explorer'},
    'R': {'command': lambda: pyautogui.hotkey('win', 'r'), 'label': 'Opening Run'},
    'W': {'command': lambda: pyautogui.hotkey('alt', 'tab'), 'label': 'Switching Apps'},
    'Q': {'command': lambda: pyautogui.hotkey('alt', 'f4'), 'label': 'Closing App'},
    
    # Web & Tabs
    'T': {'command': lambda: pyautogui.hotkey('ctrl', 't'), 'label': 'New Tab'},
    'N': {'command': lambda: pyautogui.hotkey('ctrl', 'w'), 'label': 'Closing Tab'},
    'B': {'command': lambda: pyautogui.hotkey('ctrl', 'shift', 't'), 'label': 'Restoring Tab'},
    '0': {'command': lambda: pyautogui.press('f5'), 'label': 'Refreshing...'},
    
    # Text Formatting
    '1': {'command': lambda: pyautogui.hotkey('ctrl', 'b'), 'label': 'Bold Text'},
    '2': {'command': lambda: pyautogui.hotkey('ctrl', 'u'), 'label': 'Underline Text'},
    '3': {'command': lambda: pyautogui.hotkey('ctrl', 'i'), 'label': 'Italic Text'}
}

# Tracking variables
last_executed_gesture = None
display_text = ""
text_expiry = 0

# 3. MEDIAPIPE TASKS INITIALIZATION
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

cap = cv2.VideoCapture(0)

print("--- System Live. Press 'q' to quit ---")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        timestamp = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            data_aux = []
            x_coords = [lm.x for lm in hand]
            y_coords = [lm.y for lm in hand]
    
            min_x, min_y = min(x_coords), min(y_coords)
            max_dim = max(max(x_coords) - min_x, max(y_coords) - min_y) 

            for lm in hand:
                data_aux.append((lm.x - min_x) / max_dim)
                data_aux.append((lm.y - min_y) / max_dim)

            # --- MODEL PREDICTION ---
            input_data = np.array([data_aux], dtype=np.float32)
            preds = model(input_data, training=False)
            predicted_index = np.argmax(preds)
            confidence = np.max(preds)
            last_prediction = labels[predicted_index]

            # --- UI: SHOW SCANNING STATUS ---
            color = (0, 255, 0) if confidence > 0.95 else (0, 0, 255)
            cv2.putText(frame, f"Detecting: {last_prediction} ({int(confidence*100)}%)", 
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # --- SHORTCUT EXECUTION ---
            if confidence > 0.98:
                if last_prediction in actions:
                    action_info = actions[last_prediction]
                    
                    if last_prediction != last_executed_gesture:
                        print(f"Action: {action_info['label']}")
                        action_info['command']()
                        
                        display_text = action_info['label']
                        text_expiry = time.time() + 2
                        last_executed_gesture = last_prediction
        else:
            last_executed_gesture = None
            cv2.putText(frame, "Waiting for hand...", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        hold_counter = 0
        hold_threshold = 15  # Adjust this: 10 is fast, 30 is slow/very stable
        current_active_sign = None

        # 4. DRAW THE STATUS ACTION POP-UP
        if time.time() < text_expiry:
            cv2.rectangle(frame, (10, 420), (400, 470), (0, 0, 0), -1)
            cv2.putText(frame, f"Executed: {display_text}", (20, 455), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow('ISL Shortcut Controller', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()