import cv2

def get_camera(source=0, resolutions=[(640, 480), (1280, 720)]):
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("[CAMERA] Error: Could not open camera.")
        return cap

    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if actual_w >= width * 0.9 and actual_h >= height * 0.9:
            print(f"[CAMERA] Initialized at {int(actual_w)}x{int(actual_h)}")
            return cap
            
    final_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    final_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[CAMERA] Fallback resolution used: {int(final_w)}x{int(final_h)}")
    
    return cap
