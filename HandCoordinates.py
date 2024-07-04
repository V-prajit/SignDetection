import cv2
import mediapipe as mp
import numpy as np


#does this mean we want to have the directional information of the vectors as well or do we only want the unit vecotr part of it


def HandCoordinates(videoDir, origin, scaling_factor, isOneHanded):

    mp_hands = mp.solutions.hands.Hands(static_image_mode = True,
                                        max_num_hands = 2,
                                        min_detection_confidence = 0.5,
                                        min_tracking_confidence = 0.5)
    
    cap = cv2.VideoCapture(videoDir)

    centroids_dom = []
    centroids_nondom = []
    l_delta = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = mp_hands.process(frame_rgb)

        frame_centroids_dom = None
        frame_centroids_nondom = None

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                cx = np.mean([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                cy = np.mean([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

                norm_cx = (cx - origin[0]) * scaling_factor
                norm_cy = (cy - origin[1]) * scaling_factor

                if hand_idx == 0:
                    frame_centroids_dom = (norm_cx, norm_cy)
                else:
                    frame_centroids_nondom = (norm_cx, norm_cy)
        
        centroids_dom.append(frame_centroids_dom if frame_centroids_dom else (np.nan, np.nan))

        if isOneHanded:
            centroids_nondom.append((np.nan, np.nan))
            l_delta.append((np.nan, np.nan))
        else:
            centroids_nondom.append((frame_centroids_nondom if frame_centroids_nondom else (np.nan, np.nan)))

            if frame_centroids_dom and frame_centroids_nondom:
                l_delta.append((frame_centroids_dom[0] - frame_centroids_nondom[0], frame_centroids_dom[1] - frame_centroids_nondom[1]))
            else:
                l_delta.append((np.nan, np.nan))


    centroids_dom_arr = np.array(centroids_dom)
    centroids_nondom_arr = np.array(centroids_nondom) if not isOneHanded else np.full_like(centroids_dom, np.nan)
    l_delta_arr = np.array(l_delta) if not isOneHanded else np.full_like(centroids_dom_arr, np.nan)


    cap.release()
    mp_hands.close()

    print( centroids_dom_arr, centroids_nondom_arr, l_delta_arr)
    return centroids_dom_arr, centroids_nondom_arr, origin, l_delta_arr
    