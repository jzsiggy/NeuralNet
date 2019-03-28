from imutils.video import VideoStream
import numpy as np 
import argparse
import imutils
import time
import cv2 
import threading


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, help = "path to Caffe 'deploy' prototext file")
ap.add_argument("-m", "--model", required = True, help = "path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help = "minimum probability to filter weak detections")
args = vars(ap.parse_args())



CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable", 
"dog", "horse", "motorbike","person","pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])




class Camera(threading.Thread):	
	
	def __init__(self, name, IP):
		super(Camera, self).__init__()
		self.name = name
		self.ip = IP
		self.pessoas = 0
		try:
			self.vs = VideoStream(self.ip).start()
			frame = self.vs.read()
			(h, w) = frame.shape[:2]
			self.sinal = True
		except:
			self.sinal = False
		
	
	def run(self):
		if self.sinal:	
			while True:
				frame = self.vs.read()
				frame = imutils.resize(frame, width = 400)


				(h, w) = frame.shape[:2]
				blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 0.007843, (300,300), 127.5)

				net.setInput(blob)
				detections = net.forward()

				count = 0	#Quantas pessoas tem na tela

				for i in np.arange(0, detections.shape[2]):
					confidence = detections[0,0,i,2]

					if confidence > args["confidence"]:
						idx = int(detections[0,0,i,1])
						if (idx == 15):
							count+=1
	#			print(count)
				self.pessoas = count


	def tem_sinal(self):
		try:
			self.vs = VideoStream(self.ip).start()
			frame = self.vs.read()
			(h, w) = frame.shape[:2]
			self.sinal = True
		except:
			self.sinal = False
	

	def tem_gente(self):
		if self.sinal:
#			self.quantas_pessoas()
			if self.pessoas != 0:
				self.estado = True
			else:
				self.estado = False
		else:
			self.estado = None
	

# --------------------------------XXxxXX--------------------------------


def output(*args):
	for arg in args:
		arg.tem_gente()	
		print((arg.name) + ': ' + str(arg.estado))


camera0 = Camera('camera0', 0)
camera1 = Camera('camera1', 'http://192.168.15.31:8888/video')
#camera2 = Camera('Camera2', 'IP')

camera0.start()
camera1.start()

while True:
	time.sleep(0.1)
	output(camera0)

#timeout
#	if key == ord("q"):
#		break 



#if camera.sinal == False, check every 60 seconds to see camera.sinal
#Show camera identifications on thread, to see where errors occur.
#break on 'q'
