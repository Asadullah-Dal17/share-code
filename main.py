import cv2 as cv 
from modules import gestures

gesture_recognizer = gestures.GestureRecognizer(model_path='models/gesture_recognizer.task', confidence_threshold=0.5)    

cap = cv.VideoCapture(0)
# Set camera resolution
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)  # Width
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720) # Height

# Print actual resolution to verify
width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution: {width}x{height}")

while True:
    ret, frame = cap.read()
    
    if ret is False:
        break
    frame, gestures = gesture_recognizer.recognize_gesture(frame)
    if gestures:
        print("Detected Gestures:", gestures)
    cv.imshow("frame", frame)
    key = cv.waitKey(1)
    if key ==ord("q"):
        break   
cap.release()
cv.destroyAllWindows()