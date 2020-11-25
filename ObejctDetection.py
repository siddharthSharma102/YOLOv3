# Before giving the image to the YOLO we need to create a blob, which is a way of extracting features.


import cv2
import numpy as np

# Load YOLO Algorithm.
net = cv2.dnn.readNet("D:/PYTHON/Resume Proj/YOLO/yolov3.weights",
                      "D:/PYTHON/Resume Proj/YOLO/yolov3.cfg")
classes = []
with open("D:/PYTHON/Resume Proj/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print((classes))

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading Image
img = cv2.imread("D:/PYTHON/Resume Proj/YOLO/Nature.jpg")
cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
# Blob
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing Info on the Screen
boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object Detected
            center_x = int(detection[0]*width)
            center_y = int(detection[1]*height)
            w = int(detection[2]*width)
            h = int(detection[3]*height)
            # Rectangle Coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

objects_detected = len(boxes)
for i in range(objects_detected):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y+30), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
