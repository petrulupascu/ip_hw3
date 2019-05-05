import cv2
import numpy as np
import imutils
import math

def import_mozaic(path):
  image_list = []
  image = cv2.imread(path, cv2.CV_8UC1)
  posx = 0
  posy = 0
  for i in range(2):
    posx = 0
    for j in range(20):
      tempimg = image[posy:posy+32, posx:posx+32]
      image_list.append(tempimg)
      posx = posx + 32
    posy = posy + 32
  if image_list == []:
    return("error reading mozaic file")
  else:
    return(image_list)

def save_image_list_as_mozaic(image_list, path):
  im1 = image_list[0]
  im2 = image_list[20]
  for i in range(1, 20):
    im1 = np.concatenate((im1, image_list[i]),  1)
  for j in range(21, 40):
    im2 = np.concatenate((im2, image_list[j]),  1)
  im3 = np.concatenate((im1, im2),  0)
  cv2.imwrite(path, im3)
  return True

def segment_images(image_list):
  segmented_images = []
  # noise removal
  kernel = np.ones((2,2),np.uint8)
  
  for im in image_list:
    blur = cv2.bilateralFilter(im,16,75,75)
    ret, thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(im,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) 
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    segmented_images.append(thresh)
  return(segmented_images)

def get_cedges(images):
  canny_edges = []
  #kernel = np.array([[1,1,0], [1, 1, 1], [0, 1, 0]], np.uint8) #remove discontinuites
  kernel = np.ones((2,2),np.uint8)
  for im in images:
    tedim = cv2.Canny(im, 100, 200)
    closing = cv2.morphologyEx(tedim,cv2.MORPH_CLOSE,kernel, iterations = 1)
    canny_edges.append(closing)

  return(canny_edges)


###features 
def measure_blob_circularity(image):
  cnts = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
  ko = 0
  for cnt in cnts:
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    if(leftmost[0] < 16 and rightmost[0] > 16 and topmost[1] < 16 and bottommost[1] > 16 ):
      peri = cv2.arcLength(cnt, True)
      area = cv2.contourArea(cnt)
      circularity = (4.0*math.pi*area)/(peri**2)
  #cv2.waitKey(0)
  return (circularity, area)


input_path = "./../input/LungNoduleSamples-20x2.png"
output_path_mozaic = "./../output/thresholded_mosaic.png"
edges_path_mozaic = "./../output/edges_mozaic.png"
ImageList = import_mozaic(input_path)
SegementedImageList = segment_images(ImageList)
save_image_list_as_mozaic(SegementedImageList, output_path_mozaic)

CannyEdgesImageList = get_cedges(SegementedImageList)
save_image_list_as_mozaic(CannyEdgesImageList, edges_path_mozaic)

for eci in SegementedImageList:
  cv2.imshow('imafe', eci)
  #cv2.waitKey(0)
  print(measure_blob_circularity(eci))




