import cv2
import time
import numpy as np
from math import sqrt

default_config = {
    "model": "model/yolov3.weights",
    "config": "model/yolov3.cfg",
    "categories": "model/yolov3.categories",
}

# example pic
images = "imgs/picture"

# OpenCV use BGR layout instead of RGB
normalColor = (0, 255, 0)
alertColor = (0, 0, 255)


class MarkedObject:
    def __init__(self, class_id, x1, y1, x2, y2, category):
        self.class_id = int(class_id)
        self.x1 = float(x1)
        self.x2 = float(x2)
        self.y1 = float(y1)
        self.y2 = float(y2)
        self.category = category

    def __str__(self):
        return "({}, {}) ({}, {})".format(self.x1, self.y1, self.x2, self.x2)

    def get_position(self):
        return self.x1 + self.x2 + self.y1 + self.y2

    def get_cords(self):
        return self.x1, self.y1, self.x2, self.y2

    def get_center(self):
        x = .5*(self.x2 + self.x1)
        y = .5*(self.y2 + self.y1)
        return x, y

    def get_euclidean_norms(self, other):
        # print(f"this: {self}, other {other}")
        if isinstance(other, MarkedObject):
            tx, ty = self.get_center()
            ox, oy = other.get_center()
            dx = abs(tx - ox)
            dy = abs(ty - oy)
            return dx, dy, sqrt(dx**2 + dy**2)
        else:
            raise TypeError("Provided type must be derived from MarkedObject")


class Detector(object):

    def __init__(self):
        self.classes = None
        self.categories = None
        self.net = None
        self.frame = None
        self.initialize_neural_network(**default_config)
        self.videoCapture = None
        self.videoCapture = cv2.VideoCapture("imgs/Wildlife.wmv")
        # self.videoCapture = cv2.VideoCapture("imgs/komp_1.webm")
        # self.videoCapture = cv2.VideoCapture("imgs/run.mp4")d


    def initialize_neural_network(self, model, config, categories):
        self.classes = []
        self.categories = []
        self.net = cv2.dnn.readNet(model, config)
        with open(categories, 'r') as f:
            for line in f.readlines():
                line = line.split(",")
                self.classes.append(line.pop(0))
                cat = ""
                if line:
                    cat = str(line.pop()).strip("\n ")
                    if cat:
                        print("New category: {}".format(cat))
                self.categories.append(cat)

    def draw_enclosing_frame(self, class_id, x, y, x_plus_w, y_plus_h, ismoved=False):
        label = str(self.classes[class_id])
        color = alertColor if ismoved else normalColor
        cv2.rectangle(self.frame, (x, y), (x_plus_w, y_plus_h), color, 1)
        cv2.putText(self.frame, label, (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)

    def process_next_frame(self):
        if self.videoCapture:
            for i in range(0, 5):
                self.videoCapture.read()

            retval, self.frame = self.videoCapture.read()
            if not retval:
                raise Exception("Video Capture error")
            return self.detect_objects(self.frame)

    def detect_objects_in_pic(self, index):
        self.frame = cv2.imread(images + str(index) + ".jpg")
        return self.detect_objects(self.frame)

    def detect_objects(self, frame):
        s = time.time()
        height, width, _ = frame.shape
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # get left upper corner coordinates
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        objects = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = map(int, box)
            if self.categories[i]:
                print(f"{self.categories[i]}")
            objects.append(MarkedObject(class_ids[i], x, y, x + w, y + h, self.categories[i]))
        e = time.time()
        print("\nstep duration {}".format(e - s))
        return objects

    def get_frame_shape(self):
        height, width, _ = self.frame.shape
        return width, height

    def show_frame(self):
        cv2.imshow("object detection", self.frame)
        cv2.waitKey(1)
        _DEBUG = False
        if _DEBUG:
            cv2.imwrite("object-detection.jpg", self.frame)
            cv2.destroyAllWindows()

    def __del__(self):
        if self.videoCapture:
            self.videoCapture.release()
        cv2.destroyAllWindows()
