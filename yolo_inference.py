from ultralytics import YOLO
import os

custom_path = 'detection_results' 

# Create directory if it doesn't exist
os.makedirs(custom_path, exist_ok=True)

# Load model
model = YOLO('yolov8m')

# Predict with custom save location
result = model.predict(
    source='input_videos/input_video.mp4',
    save=True,
    project=custom_path,    
    save_dir=os.path.join(custom_path)
)

print(result)
print("boxes: ")
for box in result[0].boxes:
    print(box)  