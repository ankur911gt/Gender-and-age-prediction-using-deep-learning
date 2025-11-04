# Gender and Age Detection using Deep Learning

A deep learning project that detects faces in images/video and predicts gender and age using pre-trained Caffe models and OpenCV.

## Features

- **Face Detection**: Uses OpenCV DNN face detector
- **Gender Classification**: Predicts Male/Female with confidence scores
- **Age Estimation**: Predicts age ranges (0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- **Improved Accuracy**: Enhanced preprocessing and prediction logic for better results
- **Confidence Thresholding**: Only displays high-confidence predictions
- **Image & Video Support**: Works with both image files and webcam

## Requirements

- Python 3.x
- OpenCV (opencv-python)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Files

This project uses pre-trained models:
- `opencv_face_detector_uint8.pb` - Face detection model
- `gender_net.caffemodel` - Gender classification model
- `age_net.caffemodel` - Age estimation model

**Note**: Model files are large (44MB each) and should be downloaded separately or added to Git LFS.

## Usage

### Process an image:
```bash
python gad_improved.py --image path/to/image.jpg
```

### Use webcam:
```bash
python gad_improved.py
```

### With custom confidence thresholds:
```bash
python gad_improved.py --image image.jpg --face-conf 0.8 --gender-conf 0.7 --age-conf 0.6
```

### With temporal smoothing (for video):
```bash
python gad_improved.py --smooth
```

## Arguments

- `--image`: Path to image file (if not provided, uses webcam)
- `--face-conf`: Face detection confidence threshold (default: 0.7)
- `--gender-conf`: Gender prediction confidence threshold (default: 0.6)
- `--age-conf`: Age prediction confidence threshold (default: 0.5)
- `--smooth`: Enable temporal smoothing for video streams

## Files

- `gad_improved.py` - Enhanced version with improved accuracy
- `gad.py` - Original simple version
- `requirements.txt` - Python dependencies

## Accuracy Improvements

The improved version includes:
- Better face preprocessing (preserves aging features)
- Improved age prediction logic (considers multiple top predictions)
- Confidence-based filtering
- Quality checks for face images
- Non-Maximum Suppression for face detection

## Notes

- Model files are large and may need to be downloaded separately
- CPU-only processing (GPU support can be added if needed)
- Works best with clear, front-facing faces
