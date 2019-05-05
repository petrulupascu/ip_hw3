import cv2
import numpy as np


def extract_blob_size(image):
  weight = 0
  for pval in image:
    if(pval == 255):
      weight = weight + 1
  print(weight)
  return weight


def measureCircles(image):
  im2, contours, hierarchy = cv2.findContours(image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  cv2.imshow("image", im2)
  return 1