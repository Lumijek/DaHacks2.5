import time
import sys
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import threading
from ultralytics import YOLO
import math
import os


class Camera:
	def __init__(self):
		self.in_progress = []
		self.video_capture = cv2.VideoCapture(0)
		self.w = 1280
		self.h = 720
		self.video_capture.set(3, self.w)
		self.video_capture.set(4, self.h)
		self.process_this_frame = True
		self.known_face_encodings = []
		self.known_face_names = []
		self.model = YOLO("../YOLO Weights/yolov8n.pt")
		with open('classes.txt') as f:
			self.classes = f.read().splitlines()
		self.display_faces = True

	def main_loop(self):
		while True:
			ret, frame = self.video_capture.read()
			results = self.model(frame, stream=True, verbose=False)

			if self.process_this_frame and self.display_faces:

				small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

				rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

				face_locations = face_recognition.face_locations(rgb_small_frame)


			self.process_this_frame = not self.process_this_frame
			if self.display_faces:
				for top, right, bottom, left in face_locations:
					top *= 4
					right *= 4
					bottom *= 4
					left *= 4
					midpoint = left + ((right - left) / 2)
					if (midpoint / self.w > 0.66):
						side = "left"
					elif (midpoint / self.w > 0.33):
						side = "middle"
					else:
						side = "right"
					cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

					cv2.rectangle(
						frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
					)
					font = cv2.FONT_HERSHEY_DUPLEX
					cv2.putText(
						frame, "Face", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
					)
			for r in results:
				boxes = r.boxes
				for box in boxes:
					object_cls = box.cls[0]
					name = self.classes[int(object_cls)]
					x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
					midpoint = x1 + ((x2 - x1) / 2)
					if (midpoint / self.w > 0.66):
						side = "left"
					elif (midpoint / self.w > 0.33):
						side = "middle"
					else:
						side = "right"
					cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=1)

					confidence = math.ceil((box.conf[0] * 100)) / 100


					cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, 50)
			cv2.imshow("Video", frame)

			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

		self.save_data()
		self.video_capture.release()
		cv2.destroyAllWindows()

n = Camera()
n.main_loop()