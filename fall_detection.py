import cv2
import cvzone
import math
from ultralytics import YOLO
import telepot
import os  # For optional image deletion
import time  # For time-based functionalities

# Telegram bot information (replace with your details)
token = '###'
receiver_id = '###'


def send_telegram_alert(message):
  """
  This function sends an alert message to Telegram.
  """
  try:
    bot = telepot.Bot(token)
    bot.sendMessage(receiver_id, message)
  except Exception as e:
    print(f"Error sending Telegram alert: {e}")


def fall_detection(video_path):
  """
  This function performs fall detection on the provided video path.
  """
  # cap = cv2.VideoCapture(1)

  model = YOLO("yolov8s.pt")

  classnames = []
  with open('classes.txt', 'r') as f:
      classnames = f.read().splitlines()

  fall_message_sent = False  # Flag to track if fall message is sent

  while True:
      ret, frame = cap.read()
      if not ret:  # Handle potential errors
          break

      frame = cv2.resize(frame, (980,740))

      results = model(frame)

      for info in results:
          parameters = info.boxes
          for box in parameters:
              x1, y1, x2, y2 = box.xyxy[0]
              x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
              confidence = box.conf[0]
              class_detect = box.cls[0]
              class_detect = int(class_detect)
              class_detect = classnames[class_detect]
              conf = math.ceil(confidence * 100)

              # Implement fall detection using the coordinates x1,y1,x2,y2
              height = y2 - y1
              width = x2 - x1
              threshold = height - width  # adjust threshold as needed

              if conf > 80 and class_detect == 'person':
                  cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                  cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                  if threshold < 0 and not fall_message_sent:  # Check fall and flag
                      cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)
                      # Capture image for Telegram alert (optional)
                      # ... (existing code for capturing and saving image, if needed)
                      send_telegram_alert("Fall Detected!")
                      fall_message_sent = True  # Set flag to True after sending

   
      # current_time = time.time()
      # if current_time - start_time > 10:  # Reset flag after 10 seconds
      #     fall_message_sent = False
      #     start_time = current_time  # Update start time for next fall

      cv2.imshow('frame', frame)

      # Press 'q' to quit
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

  cap.release()
  cv2.destroyAllWindows()


# Example Usage (replace with your video path)
video_path = 'fall.mp4'
fall_detection(video_path)
