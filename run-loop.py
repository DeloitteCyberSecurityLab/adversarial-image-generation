#
# (c) 2021 Deloitte Thomatsu Cyber LLC
# MIT LICENSE
# 
# Usage: python run-loop.py TARGET_CLASS
#

from sys import argv,exit
import json
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from imageai.Detection import ObjectDetection

import os
import pathlib
from sys import exit
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

# Set the path to your coco dataset here
COCO_DIR = "../../datasets/raw/coco/"
ANNOTATION_FILE = "../../datasets/raw/coco/annotations/instances_train2017.json"
TARGET_CLASS = int(argv[1])
ITERATIONS_PER_CLASS = 100 # How many targets to try
ITERATIONS_PER_INTERFERENCE = 1000 # How many merges to process per target
# CAUTION: The two iterations multiply. If you set one or both to high values, the script will run for a long time.

categories = [{"supercategory": "person","id": 1,"name": "person"},{"supercategory": "vehicle","id": 2,"name": "bicycle"},{"supercategory": "vehicle","id": 3,"name": "car"},{"supercategory": "vehicle","id": 4,"name": "motorcycle"},{"supercategory": "vehicle","id": 5,"name": "airplane"},{"supercategory": "vehicle","id": 6,"name": "bus"},{"supercategory": "vehicle","id": 7,"name": "train"},{"supercategory": "vehicle","id": 8,"name": "truck"},{"supercategory": "vehicle","id": 9,"name": "boat"},{"supercategory": "outdoor","id": 10,"name": "traffic light"},{"supercategory": "outdoor","id": 11,"name": "fire hydrant"},{"supercategory": "outdoor","id": 13,"name": "stop sign"},{"supercategory": "outdoor","id": 14,"name": "parking meter"},{"supercategory": "outdoor","id": 15,"name": "bench"},{"supercategory": "animal","id": 16,"name": "bird"},{"supercategory": "animal","id": 17,"name": "cat"},{"supercategory": "animal","id": 18,"name": "dog"},{"supercategory": "animal","id": 19,"name": "horse"},{"supercategory": "animal","id": 20,"name": "sheep"},{"supercategory": "animal","id": 21,"name": "cow"},{"supercategory": "animal","id": 22,"name": "elephant"},{"supercategory": "animal","id": 23,"name": "bear"},{"supercategory": "animal","id": 24,"name": "zebra"},{"supercategory": "animal","id": 25,"name": "giraffe"},{"supercategory": "accessory","id": 27,"name": "backpack"},{"supercategory": "accessory","id": 28,"name": "umbrella"},{"supercategory": "accessory","id": 31,"name": "handbag"},{"supercategory": "accessory","id": 32,"name": "tie"},{"supercategory": "accessory","id": 33,"name": "suitcase"},{"supercategory": "sports","id": 34,"name": "frisbee"},{"supercategory": "sports","id": 35,"name": "skis"},{"supercategory": "sports","id": 36,"name": "snowboard"},{"supercategory": "sports","id": 37,"name": "sports ball"},{"supercategory": "sports","id": 38,"name": "kite"},{"supercategory": "sports","id": 39,"name": "baseball bat"},{"supercategory": "sports","id": 40,"name": "baseball glove"},{"supercategory": "sports","id": 41,"name": "skateboard"},{"supercategory": "sports","id": 42,"name": "surfboard"},{"supercategory": "sports","id": 43,"name": "tennis racket"},{"supercategory": "kitchen","id": 44,"name": "bottle"},{"supercategory": "kitchen","id": 46,"name": "wine glass"},{"supercategory": "kitchen","id": 47,"name": "cup"},{"supercategory": "kitchen","id": 48,"name": "fork"},{"supercategory": "kitchen","id": 49,"name": "knife"},{"supercategory": "kitchen","id": 50,"name": "spoon"},{"supercategory": "kitchen","id": 51,"name": "bowl"},{"supercategory": "food","id": 52,"name": "banana"},{"supercategory": "food","id": 53,"name": "apple"},{"supercategory": "food","id": 54,"name": "sandwich"},{"supercategory": "food","id": 55,"name": "orange"},{"supercategory": "food","id": 56,"name": "broccoli"},{"supercategory": "food","id": 57,"name": "carrot"},{"supercategory": "food","id": 58,"name": "hot dog"},{"supercategory": "food","id": 59,"name": "pizza"},{"supercategory": "food","id": 60,"name": "donut"},{"supercategory": "food","id": 61,"name": "cake"},{"supercategory": "furniture","id": 62,"name": "chair"},{"supercategory": "furniture","id": 63,"name": "couch"},{"supercategory": "furniture","id": 64,"name": "potted plant"},{"supercategory": "furniture","id": 65,"name": "bed"},{"supercategory": "furniture","id": 67,"name": "dining table"},{"supercategory": "furniture","id": 70,"name": "toilet"},{"supercategory": "electronic","id": 72,"name": "tv"},{"supercategory": "electronic","id": 73,"name": "laptop"},{"supercategory": "electronic","id": 74,"name": "mouse"},{"supercategory": "electronic","id": 75,"name": "remote"},{"supercategory": "electronic","id": 76,"name": "keyboard"},{"supercategory": "electronic","id": 77,"name": "cell phone"},{"supercategory": "appliance","id": 78,"name": "microwave"},{"supercategory": "appliance","id": 79,"name": "oven"},{"supercategory": "appliance","id": 80,"name": "toaster"},{"supercategory": "appliance","id": 81,"name": "sink"},{"supercategory": "appliance","id": 82,"name": "refrigerator"},{"supercategory": "indoor","id": 84,"name": "book"},{"supercategory": "indoor","id": 85,"name": "clock"},{"supercategory": "indoor","id": 86,"name": "vase"},{"supercategory": "indoor","id": 87,"name": "scissors"},{"supercategory": "indoor","id": 88,"name": "teddy bear"},{"supercategory": "indoor","id": 89,"name": "hair drier"},{"supercategory": "indoor","id": 90,"name": "toothbrush"}]

def get_string_for_category_id(n):
    for c in categories:
        if c["id"] == n:
            return c["name"]
    return False


print ("")
print ("== INFO ==")
print ("Running with the following parameters:")
print ("Coco Directory: %s" % COCO_DIR)
print ("Correlations: %s" % ANNOTATION_FILE)
print ("Target Class: %s (%s)" % (TARGET_CLASS, get_string_for_category_id(TARGET_CLASS)))

print("")
print("== CLASS CORELLATIONS ==")

data = json.loads(open(ANNOTATION_FILE).read())
categories = data["categories"]
annotations = data["annotations"]
mapping = {}
for a in annotations:
    try:
        mapping[a["image_id"]].append(a["category_id"])
    except:
        mapping[a["image_id"]] = [a["category_id"]]

category_hits = {}

for k,v in mapping.items():
    if TARGET_CLASS in v:
        for c in v:
            try:
                category_hits[c] += 1
            except:
                category_hits[c] = 1

total_hits = category_hits[TARGET_CLASS]
corr = {}

for c in categories:
    if c["id"] not in category_hits:
        category_hits[c["id"]] = 0

for k,v in category_hits.items():
    corrpercent = float(v) / float(total_hits) * 100
    corr[k] = corrpercent

def get_filename_for_image_id(n):
    for i in data["images"]:
        if n == i["id"]:
            return i["file_name"]

def get_bounding_box_for_class_in_image(image_id, object_class):
    for a in data["annotations"]:
        if a["image_id"] == image_id and a["category_id"] == object_class:
            return [round(a["bbox"][0]), round(a["bbox"][1]), round(a["bbox"][0])+round(a["bbox"][2]), round(a["bbox"][1])+round(a["bbox"][3]) ]
    return False

def get_segmentation_for_class_in_image(image_id, object_class):
    for a in data["annotations"]:
        if a["image_id"] == image_id and a["category_id"] == object_class:
            return a["segmentation"][0]
    return False

def test_low_correlation(image_id):
    for a in data["annotations"]:
        if a["image_id"] == image_id:
            if corr[a["category_id"]] < 1:
                return True
            else:
                print(corr[a["category_id"]])
    return False

print("Set Up Networks")
detector_yolo = ObjectDetection()
detector_yolo.setModelTypeAsYOLOv3()
detector_yolo.setModelPath("yolo-coco.h5")
detector_yolo.loadModel()

detector_yolo_tiny = ObjectDetection()
detector_yolo_tiny.setModelTypeAsTinyYOLOv3()
detector_yolo_tiny.setModelPath("yolo-tiny-coco.h5")
detector_yolo_tiny.loadModel()

detector_resnet = ObjectDetection()
detector_resnet.setModelTypeAsRetinaNet()
detector_resnet.setModelPath("resnet50-coco.h5")
detector_resnet.loadModel()

detectors = [detector_yolo, detector_resnet, detector_yolo_tiny]

print("")
print("== RUN LOOP ==")

print("Select Random Image from DS:")

for l in range(ITERATIONS_PER_INTERFERENCE):
    random_class_image = ""
    while 1:
        r = random.choice(list(mapping))
        if TARGET_CLASS in mapping[r]:
            random_class_image = r
            break

    print(random_class_image)
    random_class_image_filename = get_filename_for_image_id(random_class_image)
    print(random_class_image_filename)
    random_class_image_segmentation = get_segmentation_for_class_in_image(random_class_image, TARGET_CLASS)
    print(random_class_image_segmentation)
    random_class_image_boundingbox = get_bounding_box_for_class_in_image(random_class_image, TARGET_CLASS)

    random_class_image_path = "%strain2017/%s" % (COCO_DIR, random_class_image_filename)
    print(random_class_image_path)
    im = Image.open(random_class_image_path)

    print("Detecting object in target image")
    image_path = random_class_image_path

    lowest_detection_across_detectors = 100

    for detector in detectors:
        detections = detector.detectObjectsFromImage(image_path, output_image_path="tmp.jpg", minimum_percentage_probability=60)
        certainty = 0
        for detection in detections:
            if detection["name"] == get_string_for_category_id(TARGET_CLASS):
                if detection["percentage_probability"] > certainty:
                    certainty = detection["percentage_probability"]
        print("Local Max Certainty: %f" % certainty)
        if certainty < lowest_detection_across_detectors:
            lowest_detection_across_detectors = certainty


    if lowest_detection_across_detectors < 95:
        print("Class not reliably detected. Skipping.")
        continue

    mask_im = Image.new("L", im.size, 0)
    draw = ImageDraw.Draw(mask_im)
    draw.polygon(random_class_image_segmentation, fill=255)
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(2))

    xOffset = random_class_image_boundingbox[0] * -1
    yOffset = random_class_image_boundingbox[1] * -1

    for a in range(ITERATIONS_PER_CLASS):
        iid = random.choice(list(mapping))
        while not test_low_correlation(iid):
            #print("Image correlates too well. Get another.")
            iid = random.choice(list(mapping))
        i = get_filename_for_image_id (iid)
        random_interference_image_path = "%strain2017/%s" % (COCO_DIR, i)
        image = Image.open(random_interference_image_path)   
        try:
            xShift = random.randint(0, image.width - random_class_image_boundingbox[2])
            yShift = random.randint(0, image.height - random_class_image_boundingbox[3])
        except:
            print("Object too large for target image")
            continue
        
        image.paste(im, (xOffset + xShift,yOffset + yShift), mask_im)
        image.save("auto_composites/tmp.jpg")
        image_path = "auto_composites/tmp.jpg"

        highest_detection_across_detectors = 0

        d = 0
        for detector in detectors:
            detections = detector.detectObjectsFromImage(image_path, output_image_path="tmp.jpg", minimum_percentage_probability=3)
            certainty = 0
            for detection in detections:
                if detection["name"] == get_string_for_category_id(TARGET_CLASS):
                    if detection["percentage_probability"] > certainty:
                        certainty = detection["percentage_probability"]
            print("Local Max Certainty: %f" % certainty)
            if certainty > 30:
                print("Certainty Check Failed: %d" % d)
            if certainty > highest_detection_across_detectors:
                highest_detection_across_detectors = certainty
            
            d += 1

        if highest_detection_across_detectors > 30:
            print("No significant drop in detection. Skipping.")
            continue
        
        print("Initial certainty: %s / Interfered certainty: %s. Saving." % (lowest_detection_across_detectors, highest_detection_across_detectors))
        image.save("datasets/%s-%s-%s-%s-%s-%s-%s-%s-%s.jpg" % (TARGET_CLASS, random_class_image_filename, i, lowest_detection_across_detectors, highest_detection_across_detectors, xOffset,xShift,yOffset,yShift))