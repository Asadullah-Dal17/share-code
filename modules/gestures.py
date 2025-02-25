import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class GestureRecognizer:
    def __init__(self, model_path='gesture_recognizer.task', confidence_threshold=0.7):
        self.confidence_threshold = confidence_threshold

        # Initialize the gesture recognizer options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)

        # Create the gesture recognizer
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def recognize_gesture(self, frame):
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Recognize gestures
        recognition_result = self.recognizer.recognize(mp_image)

        # Draw gestures on the frame
        gestures = []
        if recognition_result.gestures:
            for gesture in recognition_result.gestures:
                gesture_name = gesture[0].category_name
                gesture_score = gesture[0].score

                # Filter gestures based on confidence threshold
                if gesture_score >= self.confidence_threshold:
                    gestures.append((gesture_name, gesture_score))
                    cv2.putText(frame, f"{gesture_name} ({gesture_score:.2f})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return frame, gestures

    def release(self):
        """
        Release resources.
        """
        self.recognizer.close()

# Example Usage
if __name__ == "__main__":
    # Initialize webcam
    cap = cv2.VideoCapture(2)

    # Create GestureRecognizer object with a confidence threshold of 0.7
    gesture_recognizer = GestureRecognizer(model_path='models/gesture_recognizer.task', confidence_threshold=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recognize gestures
        frame, gestures = gesture_recognizer.recognize_gesture(frame)

        # Display the frame
        cv2.imshow("Gesture Recognition", frame)

        # Print detected gestures and their confidence scores
        if gestures:
            print("Detected Gestures:", gestures)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    gesture_recognizer.release()
    cv2.destroyAllWindows()