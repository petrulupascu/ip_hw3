import cv2  as cv
import numpy as np

image_list = []
image = cv.imread("./../input/LungNoduleSamples-20x2.png", cv.CV_8UC1)
posx = 0
posy = 0
for i in range(2):
  posx = 0
  for j in range(20):
    tempimg = image[posy:posy+32, posx:posx+32]
    image_list.append(tempimg)
    posx = posx + 32
    cv.waitKey(0)
  posy = posy + 32
tresholded_images = []
for im in image_list:
  imname = str(i)
  cv.imshow('image', im)
  cv.waitKey(0)
  ret, th3 = cv.threshold(im,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
  #th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,31,2)
  #cv.imshow('image', th3)
  #cv.waitKey(0)
  tresholded_images.append(th3)

"""
img = image_list[0]
cv.imshow('image', img)
cv.waitKey(0)
ret, th3 = cv.threshold(img,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
#th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,31,2)

cv.imshow('image', th3)
  """
