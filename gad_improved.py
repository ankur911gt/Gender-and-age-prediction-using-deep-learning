import cv2
import math
import argparse
import numpy as np
from collections import deque

def preprocess_face(face_img, aggressive=False):
    """
    Preprocess face image for better accuracy:
    - Light histogram equalization for better contrast (preserves aging features)
    - Minimal Gaussian blur to reduce noise
    - Proper normalization
    
    Args:
        aggressive: If True, uses more aggressive preprocessing (may reduce accuracy for elderly)
    """
    if aggressive:
        # More aggressive preprocessing (original method)
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        blurred = cv2.GaussianBlur(enhanced_bgr, (3, 3), 0)
        return blurred
    else:
        # Lighter preprocessing to preserve aging features (wrinkles, texture)
        # Use minimal or no preprocessing to preserve facial texture details
        # Direct conversion without aggressive enhancement
        return face_img.copy()

def check_face_quality(face_img):
    """
    Check if face image is of sufficient quality for prediction.
    Returns True if quality is acceptable.
    """
    if face_img.size == 0:
        return False
    
    # Check minimum size
    h, w = face_img.shape[:2]
    if h < 50 or w < 50:
        return False
    
    # Check brightness (should not be too dark or too bright)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 30 or mean_brightness > 220:
        return False
    
    # Check contrast (std deviation should be reasonable)
    std_dev = np.std(gray)
    if std_dev < 15:
        return False
    
    return True

def highlightFace(net, frame, conf_threshold=0.7):
    """Enhanced face detection with NMS"""
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    confidences = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            # Ensure coordinates are valid
            x1 = max(0, min(x1, frameWidth))
            y1 = max(0, min(y1, frameHeight))
            x2 = max(0, min(x2, frameWidth))
            y2 = max(0, min(y2, frameHeight))
            
            if x2 > x1 and y2 > y1:  # Valid bounding box
                faceBoxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
    
    # Apply Non-Maximum Suppression to remove overlapping detections
    if len(faceBoxes) > 0:
        indices = cv2.dnn.NMSBoxes(faceBoxes, confidences, conf_threshold, 0.4)
        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = [faceBoxes[i] for i in indices]
            for box in filtered_boxes:
                cv2.rectangle(frameOpencvDnn, (box[0], box[1]), (box[2], box[3]), 
                             (0, 255, 0), int(round(frameHeight/150)), 8)
            return frameOpencvDnn, filtered_boxes
    
    return frameOpencvDnn, []


def predict_with_confidence(net, blob, classes):
    """Get prediction with confidence score"""
    net.setInput(blob)
    preds = net.forward()
    idx = preds[0].argmax()
    confidence = preds[0][idx]
    prediction = classes[idx]
    return prediction, confidence

def predict_age_improved(net, blob, ageList):
    """
    Improved age prediction that considers multiple top predictions.
    Uses weighted average when probabilities are close.
    """
    net.setInput(blob)
    preds = net.forward()
    probabilities = preds[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    top_probs = probabilities[top_indices]
    
    # Print all probabilities for debugging
    print("\nAge Prediction Probabilities:")
    for i, age in enumerate(ageList):
        marker = " <-- TOP" if i == top_indices[0] else ""
        print(f"  {age}: {probabilities[i]:.3f}{marker}")
    
    # If top prediction is very confident (>0.7), use it
    if top_probs[0] > 0.7:
        return ageList[top_indices[0]], top_probs[0]
    
    # If top 2 predictions are close (within 0.15), use weighted approach
    if len(top_probs) > 1 and (top_probs[0] - top_probs[1]) < 0.15:
        # Check if one of the top predictions is elderly (60-100)
        elderly_idx = ageList.index('(60-100)')
        if elderly_idx in top_indices[:2]:
            # If elderly is in top 2 and probabilities are close, prefer elderly
            elderly_prob = probabilities[elderly_idx]
            if elderly_prob > 0.3:  # At least 30% confidence
                print(f"  Using weighted prediction: preferring elderly category")
                return ageList[elderly_idx], elderly_prob
    
    # Use top prediction
    return ageList[top_indices[0]], top_probs[0]


def temporal_smoothing(predictions, max_history=5):
    """
    Apply temporal smoothing for video to reduce flickering.
    Uses majority voting over recent frames.
    """
    if len(predictions) == 0:
        return None
    
    # Count occurrences
    counts = {}
    for pred in predictions:
        counts[pred] = counts.get(pred, 0) + 1
    
    # Return most common prediction
    return max(counts, key=counts.get)


parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--face-conf', type=float, default=0.7, help='Face detection confidence threshold')
parser.add_argument('--gender-conf', type=float, default=0.6, help='Gender prediction confidence threshold')
parser.add_argument('--age-conf', type=float, default=0.5, help='Age prediction confidence threshold')
parser.add_argument('--smooth', action='store_true', help='Enable temporal smoothing for video')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Use CPU only (no GPU detection)
faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("Using CPU for inference")

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

# Check if it's an image file or webcam
is_image_file = args.image is not None

# For temporal smoothing
if args.smooth:
    gender_history = {}
    age_history = {}

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        # For image files, break after processing once
        if is_image_file:
            break
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame, args.face_conf)
    if not faceBoxes:
        print("No face detected")
        cv2.imshow("Detecting age and gender", resultImg)
        # For image files, wait for key press then exit
        if is_image_file:
            cv2.waitKey(0)
            break
        continue

    for faceBox in faceBoxes:
        # Extract face with padding
        x1, y1, x2, y2 = faceBox
        face = frame[max(0, y1-padding):
                    min(y2+padding, frame.shape[0]-1),
                    max(0, x1-padding):
                    min(x2+padding, frame.shape[1]-1)]
        
        # Check face quality
        if not check_face_quality(face):
            print("Low quality face detected, skipping...")
            continue
        
        # Preprocess face for better accuracy
        face_processed = preprocess_face(face)
        
        # Resize to model input size
        face_resized = cv2.resize(face_processed, (227, 227))
        
        # Create blob
        blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Gender prediction with confidence
        gender, gender_conf = predict_with_confidence(genderNet, blob, genderList)
        
        # Age prediction with improved method (considers multiple top predictions)
        age, age_conf = predict_age_improved(ageNet, blob, ageList)
        
        # Temporal smoothing for video
        if args.smooth:
            face_id = f"{x1}_{y1}_{x2}_{y2}"  # Simple face ID based on position
            if face_id not in gender_history:
                gender_history[face_id] = deque(maxlen=5)
                age_history[face_id] = deque(maxlen=5)
            
            gender_history[face_id].append(gender)
            age_history[face_id].append(age)
            
            gender = temporal_smoothing(list(gender_history[face_id]))
            age = temporal_smoothing(list(age_history[face_id]))
        
        # Only show predictions if confidence is above threshold
        if gender_conf >= args.gender_conf and age_conf >= args.age_conf:
            # Format output with confidence
            label = f'{gender} ({gender_conf:.2f}), {age} ({age_conf:.2f})'
            
            # Display on image
            cv2.putText(resultImg, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            
            print(f'Gender: {gender} (Confidence: {gender_conf:.2f})')
            print(f'Age: {age[1:-1]} years (Confidence: {age_conf:.2f})')
        else:
            label = f'Low confidence (G:{gender_conf:.2f}, A:{age_conf:.2f})'
            cv2.putText(resultImg, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            print(f'Low confidence prediction - Gender: {gender_conf:.2f}, Age: {age_conf:.2f}')
    
    cv2.imshow("Detecting age and gender", resultImg)
    
    # For image files, wait for key press then exit
    if is_image_file:
        print("\nPress any key to close the window...")
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()
video.release()
