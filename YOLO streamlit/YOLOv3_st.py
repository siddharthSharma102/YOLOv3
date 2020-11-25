import cv2
import os
import time
import streamlit as st
from PIL import Image
import numpy as np

####### Widgets ########
def select_box1(path):
    files = st.sidebar.selectbox('Files', tuple([i for i in os.listdir(path) if i[-4:] == '.jpg']))
    return files

def display_img(path, file):
    st.write('## Original Image:')
    image = Image.open(path + file)
    st.image(image, "STUDY ROOM", width = 900)

def status_bar(gap):
    my_bar = st.sidebar.progress(0)
    for percent_complete in range(100):
        time.sleep(gap)
        my_bar.progress(percent_complete + 1)

####### YOLO algorithm #######
def load_yolo(path):
    classes = []
    net = cv2.dnn.readNet(path + 'yolov3.weights', path + 'yolov3.cfg')
    with open(path + 'coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] -1] for i in net.getUnconnectedOutLayers()]
    return classes, layer_names, output_layers, net

def display_info(outs, width, height):
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return boxes, class_ids, confidences, indexes

def detection(boxes, class_ids, confidences, classes, colors, indexes):
    objects_detected = len(boxes)
    for i in range(objects_detected):
        if i in indexes:
            x, y, w, h= boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color,2)
            cv2.putText(img, label, (x, y+30), cv2.FONT_HERSHEY_PLAIN, 3, color, 2)
    return img



st.write("# YOLO v3")
st.write('')
path = st.sidebar.text_input('Path Of Image ', 'D:/')
file = select_box1(path)
if st.sidebar.button('Proceed !'):
    display_img(path, file)
    status_bar(0.05)
    classes, layer_names, output_layers, net = load_yolo('D:/PYTHON/Resume Proj/YOLO/')
    colors = np.random.uniform(0,255,size = (len(classes), 3))
    img = cv2.imread(path + file)
    cv2.resize(img, None, fx = 0.4, fy = 0.4)
    height, width, channels = img.shape
#Blob
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True)
    net.setInput(blob)
    outs = net.forward(output_layers)
#Information Display
    boxes,class_ids, confidences, indexes = display_info(outs, width, height)
#Object Detection
    img = detection(boxes, class_ids, confidences, classes, colors, indexes)
#Display Final Image
    st.write('## Objects Detected in the Image:')
    st.image(img, "STUDY ROOM", width = 900)
