import cv2 
import numpy as np

image_list = []
image = cv2.imread("./../input/LungNoduleSamples-20x2.png")
posx = 0
posy = 0
for i in range(2):
  for j in range(40):
    tempimg = image[posy:posy+32, posx:posx+32]
    image_list.append(tempimg)
    posx = posx + 32
  posy = posy + 32

for im in image_list:
  cv2.imshow('image', im)
  cv2.waitKey(0)