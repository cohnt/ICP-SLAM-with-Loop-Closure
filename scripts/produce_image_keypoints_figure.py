#######################
# Fix import issues   #
import sys            #
import traceback      #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import cv2

from src.loop_closure_detection import find_keypoints, deserialize_keypoints

img_fname = "./data/EECS_3/raw_images/image0.png"
img = cv2.imread(img_fname, 0)

kp, des = deserialize_keypoints(find_keypoints(img))
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

cv2.imwrite("results/raw.png", img)
cv2.imwrite("results/keypoints.png", img2)