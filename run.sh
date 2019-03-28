#! /bin/sh

FILE=NN_detector.py

python3 $FILE \
	--prototxt MobileNetSSD_deploy.prototxt.txt \
	--model MobileNetSSD_deploy.caffemodel \
	--confidence 0.40

