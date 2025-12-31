import cv2

print("Testing camera indices...")

for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i}: Available")
        ret, frame = cap.read()
        if ret:
            print(f"  - Successfully read frame: {frame.shape}")
        cap.release()
    else:
        print(f"Camera {i}: Not available")

print("\nDone testing")