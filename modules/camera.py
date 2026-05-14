import cv2

def get_camera(source=0, resolutions=[(640, 480), (1280, 720)]):
    """
    Initializes the camera and sets a lower resolution (640x480 prioritized)
    to ensure smooth and lag-free real-time face detection.
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("[CAMERA] Error: Could not open camera.")
        return cap

    # Try setting resolutions from highest to lowest
    for width, height in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Read back actual resolution (some cameras silently ignore set requests)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # If the resolution is somewhat close to requested (accounting for slight aspect ratio differences)
        if actual_w >= width * 0.9 and actual_h >= height * 0.9:
            print(f"[CAMERA] Initialized at {int(actual_w)}x{int(actual_h)}")
            return cap
            
    # If none of the specific ones stuck, print whatever it ended up with
    final_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    final_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[CAMERA] Fallback resolution used: {int(final_w)}x{int(final_h)}")
    
    return cap
