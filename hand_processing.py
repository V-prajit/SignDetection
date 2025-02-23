import cv2
import numpy as np

def preprocess_hand_image(hand_img):
    if hand_img is None or (isinstance(hand_img, np.ndarray) and hand_img.size == 0):
        return np.zeros((50, 50), dtype=np.float32)
        
    try:
        if not isinstance(hand_img, np.ndarray):
            hand_img = np.array(hand_img)
            
        hsv = cv2.cvtColor(hand_img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        hand_img = hand_img.copy()
        hand_img[skin_mask == 0] = 0
        
        gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
        
        if gray.std() != 0:
            normalized = (gray - gray.mean()) / gray.std()
        else:
            normalized = gray - gray.mean()

        target_size = 50
        h, w = normalized.shape
        scale = target_size / max(h, w)
        scaled = cv2.resize(normalized, None, fx=scale, fy=scale)
        
        padded = np.zeros((target_size, target_size), dtype=np.float32)
        y_offset = (target_size - scaled.shape[0]) // 2
        x_offset = (target_size - scaled.shape[1]) // 2
        padded[y_offset:y_offset+scaled.shape[0], 
               x_offset:x_offset+scaled.shape[1]] = scaled
        
        return padded
        
    except Exception as e:
        print(f"Error preprocessing hand image: {str(e)}")
        return np.zeros((50, 50), dtype=np.float32)

def extract_hand_image(frame, bbox):
    try:
        if frame is None:
            return None
            
        x, y, w, h = bbox
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        return frame[y:y+h, x:x+w].copy()
    except Exception as e:
        print(f"Error extracting hand image: {str(e)}")
        return None