import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks.python.vision import GestureRecognizerResult


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest_gesture_recognition_result = None
latest_timestamp_ms = 0
gesture = None 

# Images before resizing
BACKGROUND_IMAGES_MAP = {
    "default": "image.png",
    "Thumb_Up": "backgrounds/thumb_up_bg.png", 
    "Victory": "backgrounds/victory_bg.png",   
    "Closed_Fist": "backgrounds/closed_fist_bg.png", 
    "Open_Palm": "backgrounds/peaceful_bg.png",
    "Pointing_Up":"backgrounds/pointing_up_bg.png",
    "ILoveYou":"backgrounds/cyper_punk_bg.png"
   
}

# This will store the loaded and resized background images
loaded_backgrounds = {}

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_gesture_recognition_result, latest_timestamp_ms, gesture
    latest_gesture_recognition_result = result
    latest_timestamp_ms = timestamp_ms

    # Update the global 'gesture' variable based on the latest recognition
    if result.gestures and result.gestures[0]:
        # Get the first gesture of the first detected hand
        gesture = result.gestures[0][0].category_name
        # print(f"Detected Gesture: {gesture}") 
    else:
        gesture = None # No gesture detected, reset to None
        # print("No gestures detected.") 


options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)


CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0


def segment_person_from_video(input_video_path=None, output_video_path=None):
    print(f"Loading YOLOv8 segmentation model...")
    model = YOLO('yolov8n-seg.pt')
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Frame Height :{frame_height}, Frame Width :{frame_width}")

    # --- Load and resize all background images ONCE ---
    for bg_name, bg_path in BACKGROUND_IMAGES_MAP.items():
        try:
            current_bg_image = cv2.imread(bg_path)
            if current_bg_image is None:
                raise FileNotFoundError(f"Background image not found at: {bg_path}")
            
            loaded_backgrounds[bg_name] = cv2.resize(current_bg_image, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            print(f"Loaded and resized background: {bg_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}. Please ensure background image path is correct.")
            print(f"Skipping background '{bg_name}'. Using default or previous if available.")
            # Ensure 'default' is always available even if others fail
            if bg_name == "default":
                 return # Critical error if default background is missing
        except Exception as e:
            print(f"An error occurred loading background {bg_path}: {e}")
            if bg_name == "default":
                return # Critical error if default background is missing


    frame_count = 0

    with GestureRecognizer.create_from_options(options) as recognizer:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, exiting...")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            recognizer.recognize_async(mp_image, timestamp_ms)

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"    Processing frame {frame_count}...")

            # --- Logic for displaying frames based on gesture ---
            current_output_frame = frame.copy() # Default to original frame

            if gesture == "Closed_Fist":
                # If Closed fist, display the original camera feed without segmentation
                current_output_frame = frame.copy()
            else:
                # Determine which background to use for segmentation
                active_background = loaded_backgrounds.get(gesture, loaded_backgrounds["default"])
                
                # Perform segmentation for all other gestures (including None/unrecognized)
                results = model(frame, conf=CONFIDENCE_THRESHOLD, classes=PERSON_CLASS_ID, verbose=False)
                
                # Start with the selected background
                current_output_frame = active_background.copy()

                if results[0].masks is not None:
                    combined_person_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

                    for i, mask_data in enumerate(results[0].masks.data):
                        mask = mask_data.cpu().numpy()
                        mask = cv2.resize(mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)
                        mask = (mask > 0.5).astype(np.uint8) * 255 
                        combined_person_mask = cv2.bitwise_or(combined_person_mask, mask)

                    person_mask_3_channel = cv2.merge([combined_person_mask, combined_person_mask, combined_person_mask])
                    person_segment = cv2.bitwise_and(frame, person_mask_3_channel)
                    inverse_person_mask = cv2.bitwise_not(combined_person_mask)
                    inverse_person_mask_3_channel = cv2.merge([inverse_person_mask, inverse_person_mask, inverse_person_mask])
                    
                    # Apply the inverse mask to the *active_background*
                    background_segment = cv2.bitwise_and(active_background, inverse_person_mask_3_channel)
                    
                    # Combine the person segment and the background segment
                    current_output_frame = cv2.add(person_segment, background_segment)
                # If no masks detected, current_output_frame will remain the active_background, which is desired.
            
            
            
            
            
            (text_width, text_height), baseline = cv2.getTextSize(f"Gesture:{gesture}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 3)
            cv2.rectangle(current_output_frame, (18, 45 - text_height - baseline - 5),(18 + text_width + 5, 45), (50, 205, 50), -1)
            cv2.putText(current_output_frame, f"Gesture:{gesture}", (18 + 2, 45 - baseline - 2),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow('Segmenting with Background', current_output_frame)
            
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("Segmentation application finished.")

if __name__ == "__main__":
    segment_person_from_video()