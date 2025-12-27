# Find My Frame

**Real-Time Virtual Eyewear Try-On Application Based on Face Shape Detection**

## Project Description

**Find My Frame** is a desktop application built with Python that enables users to virtually try on eyeglasses using computer vision and simple augmented reality techniques. The application automatically detects the user's face shape, provides personalized frame recommendations, and overlays selected frames on the live webcam feed — with natural behavior when the user moves their head (tilting, turning, or nodding).

## Key Features

- **Automatic Face Shape Detection**  
  Uses a pre-trained PyTorch deep learning model (85% accuracy) to classify face shape into 5 categories: Heart, Oblong, Oval, Round, and Square.

- **Personalized Frame Recommendations**  
  Suggests 3 most suitable frames based on the detected face shape.

- **Real-Time Virtual Try-On**  
  Overlays glasses with lightweight 3D-like effects (roll, yaw, pitch simulation) for natural appearance during head movement.

- **Interactive Thumbnail Guide**  
  Displays 9 frame previews (with white background) at the bottom of the screen, complete with numbering and frame names, and highlights the active frame.

- **Manual Frame Switching**  
  Press keys **1–9** to try any frame, including those outside the recommendations.

- **Screenshot Functionality**  
  Press **Space** to save a photo with the current glasses.

## Technologies Used

- **Python 3.10+**
- **MediaPipe Face Landmarker** – high-accuracy real-time facial landmark detection
- **PyTorch** – face shape classification model (https://huggingface.co/fahd9999/face_shape_classification)
- **OpenCV** – image processing and webcam interface
- **PIL/Pillow** – custom Poppins font rendering for modern UI text

## System Requirements

- Operating System: Windows / macOS / Linux
- Active webcam
- Python 3.10 or newer

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/mfaruqi35/FindMyFrame.git
   cd FindMyFrame
   ```

2. (Recommended) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\Scripts\activate       # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the folder structure is correct:
   ```
   FindMyFrame/
   ├── frames/               # Transparent PNGs for AR overlay
   ├── frames_wbg/           # White-background versions for thumbnails
   ├── test/                 # Images for testing model (optional)
   ├── requirements.txt
   ├── README.md
   ├── app.py                # Main application code
   └── FindMyFrame.ipynb     # Build process step-by-step
   ```

## How to Run

Launch via Jupyter Notebook or Python script:

```bash
jupyter notebook
# or
python app.py
```

### In-App Usage Guide

1. Face the camera
2. Align eyes and nose with the green guide area
3. Hold position for 3 seconds → automatic face shape prediction
4. Recommended glasses will appear
5. Press **1–9** to switch frames
6. Press **Space** to save screenshot
7. Press **Q** to exit

## Folder Structure

```
FindMyFrame/
├── frames/               # Transparent PNGs for AR overlay
├── frames_wbg/           # White-background versions for thumbnails
├── test/                 # Images for testing model (optional)
├── requirements.txt
├── README.md
├── app.py                # Main application code
└── FindMyFrame.ipynb     # Build process step-by-step
```

## References

1. MediaPipe Face Landmarker Documentation – Google, 2024
2. fahd9999/face_shape_classification – Hugging Face, 2024
3. Zenni Optical. "How to Choose Glasses for Your Face Shape", 2023
4. Warby Parker. "Face Shape Guide", 2024
5. OpenCV Documentation – https://docs.opencv.org

---

**© 2025 [Muhammad Faruqi] – All Rights Reserved**
