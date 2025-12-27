import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as T
import math
import os
from IPython.display import Image as IPImage, display

MODEL_PATH = "./models/model_85_nn_.pth"
MEDIAPIPE_MODEL_PATH = "./models/face_landmarker.task"

font_path_title = "fonts/Poppins-SemiBold.ttf"
font_path_regular = "fonts/Poppins-Regular.ttf"
font_path_medium = "fonts/Poppins-Medium.ttf"
font_path_small = "fonts/Poppins-Light.ttf"

font_title = ImageFont.truetype(font_path_title, 43)
font_large = ImageFont.truetype(font_path_medium, 46)
font_regular = ImageFont.truetype(font_path_regular, 25)
font_medium = ImageFont.truetype(font_path_medium, 40) 
font_small = ImageFont.truetype(font_path_small, 20) 
font_little = ImageFont.truetype(font_path_regular, 15) 

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_landmarker = FaceLandmarker.create_from_options(options)

# === Setup Model PyTorch ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

recommendations = {
    'Heart': ['cateye.png', 'round.png', 'aviator.png'],
    'Oblong': ['aviator.png', 'wayfarer.png', 'browline.png'],
    'Oval': ['rectangle.png', 'square.png', 'aviator.png'],
    'Square': ['round.png', 'oval.png', 'aviator.png'],
    'Round': ['rectangle.png', 'square.png', 'wayfarer.png']
}

all_frames = ['aviator.png', 'browline.png', 'cateye.png', 'oval.png', 'wayfarer.png', 
              'pantos.png', 'rectangle.png', 'round.png', 'square.png']

THUMB_WIDTH = 95
THUMB_HEIGHT = 95
THUMB_SPACING = 20

frames_dict = {}
thumb_list = []

for f in all_frames:
    path_overlay = os.path.join("frames", f)
    if os.path.exists(path_overlay):
        frames_dict[f] = cv2.imread(path_overlay, cv2.IMREAD_UNCHANGED)
    else:
        frames_dict[f] = None
    
    path_thumb = os.path.join("frames_wbg", f)
    if os.path.exists(path_thumb):
        img = cv2.imread(path_thumb)
        if img is not None:
            thumb = cv2.resize(img, (THUMB_WIDTH, THUMB_HEIGHT))
            thumb_list.append(thumb)
        else:
            thumb_list.append(None)
    else:
        thumb_list.append(None)

def draw_text(frame, text, position, font, color=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def is_aligned(landmarks, guide_points, threshold=30):
    le = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    re = (int(landmarks[473].x * w), int(landmarks[473].y * h))
    no = (int(landmarks[1].x * w),   int(landmarks[1].y * h))
    
    d1 = np.linalg.norm(np.array(le) - np.array(guide_points[0]))
    d2 = np.linalg.norm(np.array(re) - np.array(guide_points[1]))
    d3 = np.linalg.norm(np.array(no) - np.array(guide_points[2]))
    return d1 < threshold and d2 < threshold and d3 < threshold

def get_eye_distance(landmarks,frame_width):
    return(landmarks[468].x - landmarks[473].x) * frame_width

def overlay_glasses(frame, glasses_png, landmarks, scale_ratio=3.0):
    if glasses_png is None or len(glasses_png.shape) < 3:
        return frame
    
    h, w = frame.shape[:2]

    left_eye = (int(landmarks[468].x * w), int(landmarks[468].y * h))
    right_eye = (int(landmarks[473].x * w), int(landmarks[473].y * h))
    bridge = (int(landmarks[6].x * w), int(landmarks[6].y * h))
    nose_tip = (int(landmarks[1].x * w), int(landmarks[1].y * h))

    eye_dist = np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    if eye_dist == 0:
        return frame
    
    # Base Scaling
    base_width = int(eye_dist * scale_ratio)
    base_height = int(glasses_png.shape[0] * base_width / glasses_png.shape[1]) + 60
    
    # Roll
    delta_y = right_eye[1] - left_eye[1]
    delta_x = right_eye[0] - left_eye[0]
    roll_angle = math.degrees(math.atan2(delta_y, delta_x))

    # Yaw
    nose_x = landmarks[1].x
    eye_center_x = (landmarks[468].x + landmarks[473].x) / 2
    yaw_val = (nose_x - eye_center_x) * -3
    yaw_scale = max(0.5, 1.0 - abs(yaw_val))
    new_width = int(base_width * yaw_scale)

    # Pitch
    nose_y = landmarks[1].y
    eye_avg_y = (landmarks[468].y + landmarks[473].y) / 2
    pitch_val = (nose_y - eye_avg_y) * 10
    pitch_scale = max(0.8, 1.0 - abs(pitch_val) * 0.5)
    new_height = int(base_height * pitch_scale)

    glasses_resized = cv2.resize(glasses_png, (new_width, new_height), interpolation=cv2.INTER_AREA)

    M = cv2.getRotationMatrix2D((new_width // 2, new_height // 2), -roll_angle, 1)
    glasses_rotated = cv2.warpAffine(glasses_resized, M, (new_width, new_height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0,0,0,0))
    
    center_x = bridge[0]
    center_y = bridge[1] + int(eye_dist * 0.05)

    x1 = center_x - glasses_rotated.shape[1] // 2
    y1 = center_y - glasses_rotated.shape[0] // 2
    x2 = x1 + glasses_rotated.shape[1]
    y2 = y1 + glasses_rotated.shape[0]
    
    x1_clip, y1_clip = max(0, x1), max(0, y1)
    x2_clip, y2_clip = max(0, x2), max(0, y2)

    target_w = x2_clip - x1_clip
    target_h = y2_clip - y1_clip
    
    if target_w <= 0 or target_h <= 0:
        return frame

    offset_x = max(0, -x1)
    offset_y = max(0, -y1)

    crop_x2 = min(offset_x + target_w, glasses_rotated.shape[1])
    crop_y2 = min(offset_y + target_h, glasses_rotated.shape[0])
    
    if offset_x >= glasses_rotated.shape[1] or offset_y >= glasses_rotated.shape[0]:
        return frame
    
    if crop_x2 <= offset_x or crop_y2 <= offset_y:
        return frame

    overlay_crop = glasses_rotated[offset_y:crop_y2, offset_x:crop_x2]
    actual_h, actual_w = overlay_crop.shape[:2]
    
    final_x2 = x1_clip + actual_w
    final_y2 = y1_clip + actual_h

    if final_x2 > w or final_y2 > h:
        available_w = min(target_w, w - x1_clip)
        available_h = min(target_h, h - y1_clip)
        
        if available_w <= 0 or available_h <= 0:
            return frame
            
        overlay_crop = cv2.resize(overlay_crop, (available_w, available_h), interpolation=cv2.INTER_AREA)
        final_x2 = x1_clip + available_w
        final_y2 = y1_clip + available_h

    if overlay_crop.shape[0] != (final_y2 - y1_clip) or overlay_crop.shape[1] != (final_x2 - x1_clip):
        return frame

    if overlay_crop.shape[2] == 4:
        try:
            alpha = overlay_crop[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            
            for c in range(3):
                frame[y1_clip:final_y2, x1_clip:final_x2, c] = (
                    alpha * overlay_crop[:, :, c] + 
                    alpha_inv * frame[y1_clip:final_y2, x1_clip:final_x2, c]
                )
        except ValueError:
            return frame
    else:
        try:
            frame[y1_clip:final_y2, x1_clip:final_x2] = overlay_crop[:, :, :3]
        except ValueError:
            return frame
    
    return frame

LEFT_EYE = 468
RIGHT_EYE = 473
NOSE_TIP = 1

cap = cv2.VideoCapture(0)
cap.set(3, 1280); cap.set(4, 720)

state = "ALIGNING"          
aligned_frames = 0
required_frames = 90        
threshold = 30
predicted_shape = None
current_glasses = None
screenshot_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h,w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.time() * 1000)
    results = face_landmarker.detect_for_video(mp_img, timestamp_ms)

    if results.face_landmarks:
        landmark = results.face_landmarks[0]

        if state == "ALIGNING":
            eye_dist = int(w * 0.065)
            eye_y_offset = int(h * 0.003)
            nose_y_offset = int(h * 0.16)
            GUIDE_LEFT_EYE = (center_x - eye_dist, center_y - eye_y_offset)
            GUIDE_RIGHT_EYE = (center_x + eye_dist, center_y - eye_y_offset)
            GUIDE_NOSE = (center_x, center_y + nose_y_offset)

            # Green Area
            for pt in [GUIDE_LEFT_EYE, GUIDE_RIGHT_EYE, GUIDE_NOSE]:
                cv2.circle(frame, pt, threshold, (92,234,20), 3)
            
            # Blue point
            left_eye = (int(landmark[LEFT_EYE].x * w), int(landmark[LEFT_EYE].y * h))
            right_eye = (int(landmark[RIGHT_EYE].x * w), int(landmark[RIGHT_EYE].y * h))
            nose = (int(landmark[NOSE_TIP].x * w),   int(landmark[NOSE_TIP].y * h))
            for p in [left_eye, right_eye, nose]:
                cv2.circle(frame, p, 8, (254,187,0), -1)
            
            # Alighment Check
            if is_aligned(landmark, [GUIDE_LEFT_EYE, GUIDE_RIGHT_EYE, GUIDE_NOSE], threshold):
                aligned_frames += 1
                frame = draw_text(frame, "Tahan Posisimu!", (center_x - 200, center_y - 150),
                                  font_large, (100, 250, 0))
                
                if aligned_frames >= required_frames and predicted_shape is None:
                    clean_frame = frame.copy()
                    cv2.imwrite("./results/final_clean_image.jpg", clean_frame)
                    print("\nScreenshot saved")

                    pil_img = Image.fromarray(cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB))
                    tensor = transform(pil_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        out = model(tensor)
                        idx = torch.argmax(out, 1).item()
                        predicted_shape = class_names[idx]
                        current_glasses = frames_dict[recommendations[predicted_shape][0]]
                    print(f"Bentuk wajah terdeteksi: {predicted_shape.upper()}")
                    state = "PREDICTED"
            else:
                aligned_frames = 0

        if state == "PREDICTED":
            if len(thumb_list) > 0:
                total_thumbs = len([t for t in thumb_list if t is not None])
                if total_thumbs == 0:
                    total_thumbs = 1 
                
                total_width = total_thumbs * THUMB_WIDTH + (total_thumbs - 1) * THUMB_SPACING
                start_x = max(50, (w - total_width) // 2) 
                start_y = h - THUMB_HEIGHT - 60 

                current_idx = 0
                for idx, thumb in enumerate(thumb_list):
                    if thumb is None:
                        continue
                    
                    x = start_x + current_idx * (THUMB_WIDTH + THUMB_SPACING)
                    y = start_y
                    
                    if x + THUMB_WIDTH > w:
                        break 
                    
                    # Background & highlight
                    cv2.rectangle(frame, (x-5, y-5), (x+THUMB_WIDTH+5, y+THUMB_HEIGHT+5), (60, 60, 60), -1)
                    if current_glasses is frames_dict[all_frames[idx]]:
                        cv2.rectangle(frame, (x-5, y-5), (x+THUMB_WIDTH+5, y+THUMB_HEIGHT+5), (0, 255, 255), 5)
                    
                    frame[y:y+THUMB_HEIGHT, x:x+THUMB_WIDTH] = thumb
                    
                    frame_name = all_frames[idx][:-4].upper()
                    text = f"({idx+1}) {frame_name}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    text_x = x + (THUMB_WIDTH - text_size[0]) // 2
                    text_y = y - 10
                    
                    frame = draw_text(frame, text, (text_x + 6, text_y - 21), font_little, (255, 255, 255))
                    
                    current_idx += 1
            if current_glasses is not None:
                frame = overlay_glasses(frame, current_glasses, landmark)

            # Teks hasil & rekomendasi
            frame = draw_text(frame, f"Bentuk Wajah: {predicted_shape.upper()}", (50, 10), font_title, (255,255,255))
            
            rec = recommendations[predicted_shape]
            frame = draw_text(frame, f"Rekomendasi: ({rec[0][:-4]}, {rec[1][:-4]}, {rec[2][:-4]})", (50, 60), font_regular, (255,255,255))
            
            frame = draw_text(frame, "1-9: Ganti Frame | Spasi: Screenshot | Q: Keluar", (50, h-40), font_small, (255,255,255))

    if state == "ALIGNING":
        frame = draw_text(frame, "Find My Frame", (50, 10), font_title, (255,255,255))
        frame = draw_text(frame, "Posisikan titik biru ke area hijau, dan tahan 3 detik", (50, 60), font_regular, (255,255,255))

    cv2.imshow("Find My Frame App ", frame)

   # Keyboard control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        screenshot_count += 1
        filename = f"./results/ar_result_{screenshot_count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot AR tersimpan: {filename}")
    elif '1' <= chr(key) <= '9':
        idx = int(chr(key)) - 1
        if idx < len(all_frames):
            current_glasses = frames_dict[all_frames[idx]]
            print(f"Ganti frame: {all_frames[idx]}")

cap.release()
cv2.destroyAllWindows()

# Tampilkan screenshot terakhir di notebook
if screenshot_count > 0:
    display(IPImage(f"ar_result_{screenshot_count}.jpg", width=700))
