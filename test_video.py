#!/usr/bin/env python2

import numpy as np
import caffe
import random
import cv2
import GPUtil
import time

DEBUG = False

alpha = 0.5

frame_count = 0
def random_rgb(num):
   colors = []
   for i in range(num):
       if i == 0:
            colors.append((0, 0, 0))
          #   colors.append((255, 255, 255))

       elif i in range(1, 5) or i == 11:
            colors.append((255, 0, 0))
       elif i in range(6, 10) or i == 12:
            colors.append((0, 255, 0))
       elif i == 5:
            colors.append((0, 0, 255))
       elif i == 10:
            colors.append((255, 255, 0))
   return colors

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

lane_colors = random_rgb(13)

cap = cv2.VideoCapture('/home/almon/personal_repos/apollo_lane/test.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.avi',fourcc, 30, (640,480))

# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    t0 = time.time()

    # load image
    # image = caffe.io.load_image('input.png')

    # resize input image

    # crop top half of image using open cv
    crop_image = frame[:, 480:, :]
    cv_resize_image = cv2.resize(frame, (640, 480))

    resize_image = caffe.io.resize_image(frame, [480, 640])

    #print(resize_image.shape)
    transformed_image = transformer.preprocess('data', resize_image)

    # lane predict
    net.blobs['data'].data[...] = transformed_image
    output = net.forward()
    mask_color = np.zeros((480, 640, 3), np.uint8)
    # make mask_color all white
    # mask_color[:, :, :] = (255, 255, 255)
    confidence_thread = 0.95


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

    # combine with original image
    dst = cv2.addWeighted(cv_resize_image, alpha, mask_color, 1.0-alpha, 0.0, dtype = cv2.CV_32F)
   
        # GPUtil.showUtilization()
    out.write(dst)

        # use cv2 to show the image
        # cv2.imshow('mask', mask_color)
    # calculate fps
    t1 = time.time()
    fps = 1.0/(t1-t0)
    print(frame_count, fps)
    frame_count += 1
    
  # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()