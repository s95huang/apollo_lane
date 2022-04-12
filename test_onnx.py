#!/usr/bin/env python3

from traceback import print_tb
import numpy as np    # we're going to use numpy to process input and output data
import onnxruntime    # to inference ONNX models, we use the ONNX Runtime
import onnx
from onnx import numpy_helper
import urllib.request
import json
import time
import cv2
import random

#define the priority order for the execution providers
# prefer CUDA Execution Provider over CPU Execution Provider
EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# load model
session = onnxruntime.InferenceSession('apollo_lane.onnx', providers=EP_list)


# session = onnxruntime.InferenceSession('apollo_lane.onnx', providers=onnxruntime.get_available_providers())


# # get the name of the first input of the model
# input_name = session.get_inputs()[0].name  

# print('Input Name:', input_name)

# output_name = session.get_outputs()[0].name
# output_name2 = session.get_outputs()[1].name
# print('Output Name:', output_name)
# print('Output Name2:', output_name2)

# read image 
image = cv2.imread('input.jpg')

# save as a numpy array
image_np = np.array(image)

resized = cv2.resize(image_np, (480,640))

#show the image
# cv2.imshow('image', resized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

image_data = np.array(resized).transpose((2,0,1))


img_data = np.array(image_data, dtype=np.float32)


# normalize
image_mean = np.array([95, 99, 96])

# subtract image mean from imgage_data
img_data[0, :, :] = img_data[0, :, :] - image_mean[0]
img_data[1, :, :] = img_data[1, :, :] - image_mean[1]
img_data[2, :, :] = img_data[2, :, :] - image_mean[2]

# scale to [0, 255]
mean_array = np.array([95, 99, 96])

print(mean_array.shape)
print(len(mean_array.shape))
print(mean_array.ndim)

#add batch channel
input_data = image_data.reshape(1, 3, 480, 640).astype('float32')



input_name = session.get_inputs()[0].name  

# print('Input Name:', input_name)
# check the shape of the input data
# print('Input Shape:', input_data.shape)

raw_result = session.run([], {input_name: input_data})

# get raw_result shape
raw_result_shape = raw_result[0].shape
raw_result_shape_2 = raw_result[1].shape

# print(raw_result_shape)
# print(raw_result_shape_2)

# plot the result
mask_color = np.zeros((480, 640, 3), np.uint8)
confidence_thread = 0.7

def random_rgb(num):
   colors = []
   for i in range(num):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((r, g, b))
   return colors

lane_colors = random_rgb(13)

for lane in enumerate(raw_result[0][0,0,:,:]):

    # index = (lane >= confidence_thread)
    # for row in range(lane.shape[0]):
    #     for col in range(lane.shape[1]):
    #         if lane[row][col] > confidence_thread:
    #             mask_color[row][col] = lane_colors[id]
    mask_color = lane_colors[0]
    # print(lane[0].shape)

# print(raw_result[0][0,0,:,:])
# use cv2 to show the image
# cv2.imshow('mask', mask_color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
