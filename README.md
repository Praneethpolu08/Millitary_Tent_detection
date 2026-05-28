# Tent Detection

This project is a Streamlit-based web application for detecting tents in satellite or aerial images using a YOLO object detection model. It is designed to help users upload an image, run inference, and inspect the detected tent locations along with simple confidence metrics.

## What this project does
- Accepts an uploaded image from the user
- Uses a trained YOLO model (`best.pt`) to detect tents
- Displays the original image and the annotated detection result side by side
- Shows basic detection metrics such as total detections and average confidence

## How it works
1. The user uploads an image through the Streamlit interface.
2. The image is converted into a NumPy array for processing.
3. The YOLO model predicts tent locations using the chosen confidence threshold.
4. The app renders the annotated image and summary information for the user.

## Project structure
- `app.py` — main Streamlit application
- `src/detector.py` — model loading and detection logic
- `src/image_utils.py` — image conversion helpers
- `src/metrics.py` — detection summary calculations
- `src/ui_components.py` — sidebar and UI components
- `best.pt` — trained detection model

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- The model file is expected at `best.pt`.
- Make sure the required dependencies are installed before running the app.
- For large model files, GitHub may warn about size limits when pushing to a remote repository.
