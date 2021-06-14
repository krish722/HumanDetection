import cv2
import argparse
import sys
import time
import imutils
import numpy as np
import os
from datetime import datetime, timedelta

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

def getCamera(url):
    while True:
        try:
            print("getting camera ",url)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            vs = cv2.VideoCapture(url)
            rs, frame = vs.read()
            if not rs:
                print("error camera no frame:")
                time.sleep(5.0)

            print("got camera at ",datetime.now())
            sys.stdout.flush()
            return vs
        except Exception as e:
            print("error camera", str(e))
            time.sleep(4.0)


def getFrame(url, vs):
    if not vs:
        vs = getCamera(url)
    while True:
        rs = vs.grab()
        if rs:
            return vs
        else:
            print("can't get at ",datetime.now())
            vs.release()
            vs = getCamera(url)

def getDetection(args, smallFrame):
    blob = cv2.dnn.blobFromImage(smallFrame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()
    trackers = []
    image = smallFrame
    found = False
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated
        # with the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by requiring a minimum
        # confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # detections list
            idx = int(detections[0, 0, i, 1])

            # if the class label is not a person, ignore it
            if CLASSES[idx] != "person":
                continue

            # compute the (x, y)-coordinates of the bounding box
            # for the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            start_point = (startX, startY) 
            end_point = (endX, endY) 
            color = (255, 0, 0) 
            thickness = 2
            found = True
            image = cv2.rectangle(image, start_point, end_point, color, thickness) 
    return found, image

def sendMessage(image):
    os.system("./imessage 'human detection' ")
    cv2.imwrite("frame1.jpg", image)
    os.system("./imessage_img1 $PWD/frame1.jpg")

ap = argparse.ArgumentParser()
ap.add_argument("-u","--url", required=True, help="#url for the video")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="#minimum probability to filter weak detections")
ap.add_argument("-s", "--skip-frames", type=int, default=10,
	help="# of skip frames between detections")
ap.add_argument("-v", "--view", required=False,  default="false",
	help="# view video")
ap.add_argument("-t", "--threshold", required=False,  default=4000,
	help="# motion detection threshold")
args = vars(ap.parse_args())
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")
W = None
H = None
totalFrames = 0
newImage = smallFrame =0
image_sent = datetime.now()
image_sent -= timedelta(minutes=1)
vs = None
found = False
avg = None
outVideo = None
while True:
    notToSkip = totalFrames % args["skip_frames"] == 0
    vs = getFrame(args["url"], vs)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    rt, frame = vs.retrieve()
    if not rt:
        totalFrames = 0
        if outVideo:
            outVideo.release()
            outVideo = None
        vs = getFrame(args["url"], vs)
        continue
    if notToSkip:
        smallFrame = imutils.resize(frame, width=500)
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2grey)
        grey = cv2.GaussianBlur(grey, (21, 21), 0)

        if avg is None or totalFrames > 600:
            totalFrames = 0
            print("[INFO] starting background model...")
            avg = grey.copy().astype("float")
            continue

        cv2.accumulateWeighted(grey, avg, 0.5)
        frameDelta = cv2.absdiff(grey, cv2.convertScaleAbs(avg))

        thresh = cv2.threshold(frameDelta, 5, 255,
            cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)

        motion=False
        for c in cnts:
            if cv2.contourArea(c) < args['threshold']:
                continue

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            motion=True
            break

        if motion:
            print("people detection....")
            totalFrames = 0
            rgb = cv2.cvtColor(smallFrame, cv2.COLOR_BGR2RGB)
            if W is None or H is None:
                (H, W) = smallFrame.shape[:2]
            found, newImage = getDetection(args, smallFrame)

    if found:
        cur = datetime.now()
        checkcur = cur - timedelta(minutes=1)
        if checkcur > image_sent:
            image_sent = cur
            sendMessage(newImage)
            width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') #opencv3.0
            timestr = time.strftime("%Y%m%d-%H%M%S.mp4")
            timestr = "HumanDetectionRecord/" + timestr
            if not outVideo:
                outVideo = cv2.VideoWriter(timestr,fourcc, 15.0, (width,height))

    if args["view"] != "false":
        cv2.imshow('frame',newImage)
        cv2.imshow('grey',grey)
    if outVideo:
        outVideo.write(frame)
        if not found and not motion:
            print("Stopping video")
            outVideo.release()
            outVideo = None
    totalFrames += 1
