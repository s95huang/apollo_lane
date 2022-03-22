#!/usr/bin/env python2

import numpy as np
import caffe
import random
import cv2
import GPUtil

DEBUG = False

# load model
model_def = 'deploy.prototxt'
model_weights = 'deploy.caffemodel'
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# define transformer
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_raw_scale('data', 255) # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
transformer.set_mean('data', np.array([95, 99, 96]))            # subtract the dataset-mean value in each channel

# load image
image = caffe.io.load_image('input.png')

# resize input image


crop_image = image[:, 312:, :]
resize_image = caffe.io.resize_image(image, [480, 640])

#print(resize_image.shape)
transformed_image = transformer.preprocess('data', resize_image)

# lane predict
net.blobs['data'].data[...] = transformed_image
output = net.forward()
mask_color = np.zeros((480, 640, 3), np.uint8)
confidence_thread = 0.3

def random_rgb(num):
   colors = []
   for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
   return colors

lane_colors = random_rgb(13)
if DEBUG:
    print(lane_colors)
    print(output['softmax'][0])
    print("lane output shape: ", output['softmax'].shape)

for id, lane in enumerate(output['softmax'][0]):
    index = (lane >= confidence_thread)
    # for row in range(lane.shape[0]):
    #     for col in range(lane.shape[1]):
    #         if lane[row][col] > confidence_thread:
    #             mask_color[row][col] = lane_colors[id]
    mask_color[lane >= confidence_thread] = lane_colors[id]

GPUtil.showUtilization()

# use cv2 to show the image
cv2.imshow('mask', mask_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

