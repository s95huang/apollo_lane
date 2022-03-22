import cv2
import sys
from caffe2.python import core, workspace, net_printer
import future
from caffe2.proto import caffe2_pb2
import numpy as np
import os


# load the model and weights
config = '/home/almon/personal_repos/apollo_lane/deploy.prototxt'
weights = '/home/almon/personal_repos/apollo_lane/deploy.caffemodel'

# check if the model exists
if config is None or weights is None:
    print('Error: model definition or weights not found')
    sys.exit()
# load the model
# p = workspace.Predictor(config, weights)
model = cv2.dnn_DetectionModel(weights, config)
# model.setInputParams(size=(640, 480), mean=(0, 0, 0), scale=1.0)

# run inference on the image
image = cv2.imread('/home/almon/personal_repos/apollo_lane/image.png')

# convert image to bgr
bgr_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# mean bgr vlaue of the image
mean_b = 95
mean_g = 99
mean_r = 96


image_blob = cv2.dnn.blobFromImage(bgr_image, 1.0, (640, 480), (mean_b, mean_g, mean_r), False, False)

model.setInput(image_blob)

# get model output
outputs = model.forward()

