import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

def extract_frames(video_path, frame_rate=1):
    """Extract frames from the video at a specified frame rate."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if frame_number % interval == 0:
            frames.append(frame)
    
    cap.release()
    return frames

def classify_frame(frame, model):
    """Classify a single frame using the pre-trained model."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    return decoded_predictions

def analyze_video(video_path):
    model = MobileNetV2(weights='imagenet')
    frames = extract_frames(video_path)
    
    issues = {
        'electric_issue': 0,
        'water_issue': 0,
        'security_issue': 0
    }
    
    for frame in frames:
        predictions = classify_frame(frame, model)
        for _, label, _ in predictions:
            if 'electric' in label.lower():
                issues['electric_issue'] += 1
            elif 'water' in label.lower():
                issues['water_issue'] += 1
            elif 'security' in label.lower():
                issues['security_issue'] += 1
    
    result = []
    for issue, count in issues.items():
        if count > len(frames) * 0.05:  
            result.append(issue)
    
    if result:
        return f"This video is related with: {', '.join(result)}"
    else:
        return "No significant issues detected."
