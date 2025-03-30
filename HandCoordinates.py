import cv2
import mediapipe as mp
import numpy as np

def HandCoordinates(videoDir, origin, scaling_factor, isOneHanded):
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1
    )
    
    cap = cv2.VideoCapture(videoDir)
    if not cap.isOpened():
        print(f"Error: Could not open video at {videoDir}")
        empty = np.array([])
        return empty, empty, empty, empty, origin, empty, empty, empty, empty

    # Get reported properties (these might be inaccurate)
    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Reported video properties: {width}x{height}, {reported_fps} fps, {reported_frame_count} frames")
    
    centroids_dom = []
    centroids_nondom = []
    bboxes_dom = []
    bboxes_nondom = []
    l_delta = []
    
    found_hand = False
    frame_count = 0
    
    # Force reading all frames regardless of reported count
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 10 == 0:  # Report progress every 10 frames
            print(f"Processing frame {frame_count}")
        
        frame_height, frame_width = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        frame_centroids_dom = None
        frame_centroids_nondom = None
        frame_bbox_dom = None
        frame_bbox_nondom = None

        if results.multi_hand_landmarks:
            hand_data = []
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
                y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
                cx = np.mean(x_coords)
                cy = np.mean(y_coords)
                
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame_width, x_max + padding)
                y_max = min(frame_height, y_max + padding)
                
                norm_cx = (cx - origin[0]) * scaling_factor
                norm_cy = (cy - origin[1]) * scaling_factor
                
                hand_data.append({
                    'centroid': (norm_cx, norm_cy),
                    'bbox': (int(x_min), int(y_min), 
                            int(x_max - x_min), int(y_max - y_min)),
                    'raw_x': cx
                })
            
            if hand_data:
                hand_data.sort(key=lambda x: x['raw_x'], reverse=True)
                
                if len(hand_data) >= 1:
                    frame_centroids_dom = hand_data[0]['centroid']
                    frame_bbox_dom = hand_data[0]['bbox']
                    found_hand = True 
                    
                    if len(hand_data) >= 2 and not isOneHanded:
                        frame_centroids_nondom = hand_data[1]['centroid']
                        frame_bbox_nondom = hand_data[1]['bbox']

        if frame_centroids_dom:
            centroids_dom.append(frame_centroids_dom)
            bboxes_dom.append(frame_bbox_dom)
        else:
            centroids_dom.append((np.nan, np.nan))
            bboxes_dom.append((0, 0, 100, 100))
        
        if isOneHanded:
            centroids_nondom.append((np.nan, np.nan))
            bboxes_nondom.append((0, 0, 100, 100))
            l_delta.append((np.nan, np.nan))
        else:
            if frame_centroids_nondom:
                centroids_nondom.append(frame_centroids_nondom)
                bboxes_nondom.append(frame_bbox_nondom)

                delta = (
                    frame_centroids_dom[0] - frame_centroids_nondom[0],
                    frame_centroids_dom[1] - frame_centroids_nondom[1]
                )
                l_delta.append(delta)
            else:
                centroids_nondom.append((np.nan, np.nan))
                bboxes_nondom.append((0, 0, 100, 100))
                l_delta.append((np.nan, np.nan))

    cap.release()
    mp_hands.close()

    print(f"Actually processed {frame_count} frames (reported: {reported_frame_count})")
    print(f"Found hands in at least one frame: {found_hand}")
    
    if frame_count <= 1:
        print("ERROR: Only processed one frame! The video might be corrupted.")
        empty = np.array([])
        return empty, empty, empty, empty, origin, empty, empty, empty, empty

    centroids_dom_arr = np.array(centroids_dom, dtype=np.float32)
    centroids_nondom_arr = np.array(centroids_nondom, dtype=np.float32)
    bboxes_dom_arr = np.array(bboxes_dom, dtype=np.int32)
    bboxes_nondom_arr = np.array(bboxes_nondom, dtype=np.int32)
    l_delta_arr = np.array(l_delta, dtype=np.float32)

    def compute_orientation(coords_arr):
        orientation_list = []
        length = len(coords_arr)
        
        if length <= 1:
            return np.array([[0, 0]] * length, dtype=np.float32)
        
        for i in range(length):
            i_prev = max(0, i-1)
            i_next = min(length-1, i+1)
            
            dx = coords_arr[i_next,0] - coords_arr[i_prev,0]
            dy = coords_arr[i_next,1] - coords_arr[i_prev,1]
            norm = np.sqrt(dx*dx + dy*dy)
            
            if norm < 1e-9 or np.isnan(norm):
                orientation_list.append([np.nan, np.nan])
            else:
                orientation_list.append([dx/norm, dy/norm])
                
        return np.array(orientation_list, dtype=np.float32)

    orientation_dom_arr = compute_orientation(centroids_dom_arr)
    orientation_nondom_arr = compute_orientation(centroids_nondom_arr)
    orientation_delta_arr = compute_orientation(l_delta_arr)

    total_frames = len(centroids_dom)
    detected_frames = np.sum(~np.isnan(centroids_dom_arr[:, 0]))
    print(f"\nHand Detection Statistics:")
    print(f"Total frames: {total_frames}")
    print(f"Frames with detected hands: {detected_frames}")
    detection_rate = (detected_frames/total_frames)*100 if total_frames > 0 else 0
    print(f"Detection rate: {detection_rate:.2f}%")

    if not found_hand or detection_rate < 5:
        print("Insufficient hand detection. Returning empty arrays.")
        empty = np.array([])
        return empty, empty, empty, empty, origin, empty, empty, empty, empty

    return (
        centroids_dom_arr,
        centroids_nondom_arr,
        bboxes_dom_arr,   
        bboxes_nondom_arr, 
        origin,
        l_delta_arr,
        orientation_dom_arr,
        orientation_nondom_arr,
        orientation_delta_arr
    )