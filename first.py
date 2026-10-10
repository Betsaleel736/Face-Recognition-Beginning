
import cv2
import tkinter as tk
from ultralytics import YOLO

# Load YOLO models
model = YOLO("yolov8n.pt")
face_models = YOLO("yolov8n-face.pt")

# Set window dimensions
WINDOW_NAME = "my window"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Get screen resolution
root = tk.Tk()
root.withdraw()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.destroy()

# Calculate the centre position
window_x = (screen_width - WINDOW_WIDTH) // 2
window_y = (screen_height - WINDOW_HEIGHT) // 2

# Create and position the window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(
    WINDOW_NAME,
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)
cv2.moveWindow(
    WINDOW_NAME,
    window_x,
    window_y
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read webcam frame.")
        break

    # Detect faces
    face_result = face_models(frame, verbose=False)

    # Detect general objects
    result = model(frame, verbose=False)

    # Draw general object detections
    ball_pers = result[0].plot()

    # Draw face detections on the same frame
    face = face_result[0].plot(img=ball_pers)

    # Display detections
    cv2.imshow(WINDOW_NAME, ball_pers)

    # Press X to exit
    if cv2.waitKey(1) & 0xFF == ord("x"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()