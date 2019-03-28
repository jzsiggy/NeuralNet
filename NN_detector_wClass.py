from imutils.video import VideoStream
import numpy as np 
import argparse
import imutils
import time
import cv2 


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




class Camera:	
	
	def __init__(self, name, IP):
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
		
	def tem_sinal(self):
		try:
			self.vs = VideoStream(self.ip).start()
			frame = self.vs.read()
			(h, w) = frame.shape[:2]
			self.sinal = True
		except:
			self.sinal = False
		print('sinal checado')

	
	def quantas_pessoas(self):	
#		print("[INFO] starting video stream...")
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

		self.pessoas = count
#		print(self.pessoas)


	def tem_gente(self):
		if self.sinal:
			self.quantas_pessoas()
			if self.pessoas != 0:
				self.estado = True
			else:
				self.estado = False
		else:
			self.estado = None


def output(*args):
	for arg in args:
		arg.tem_gente()	
		print((arg.name) + ': ' + str(arg.estado))


camera0 = Camera('camera0', 0)
camera1 = Camera('camera1', 'http://10.102.3.105:8888/video')
camera2 = Camera('Camera2', 'http://10.102.3.123:8080/video')

while True:
	time.sleep(0)
	output(camera0, camera1, camera2)

#timeout
#	if key == ord("q"):
#		break 



#thread a camera para ficar rodando => def run: loop while, etc ----------> NN_detector_Wthread FEITO
#if camera.sinal == False, check every 60 seconds to see camera.sinal
#break on 'q'





