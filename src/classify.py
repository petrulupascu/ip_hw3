import math
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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
def get_features(images):
  features = []
  for image in images:
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
    features.append((circularity, area))
  return (features)

def build_dataset(FeatureList):
  dataset = np.zeros((40, 3))
  for i in range(40):
    if i < 20:
      dataset[i][0] = 0
      dataset[i][1] = FeatureList[i][0]
      dataset[i][2] = FeatureList[i][1]
    else:
      dataset[i][0] = 1
      dataset[i][1] = FeatureList[i][0]
      dataset[i][2] = FeatureList[i][1]
  return dataset


input_path = "./../input/LungNoduleSamples-20x2.png"
output_path_mozaic = "./../output/thresholded_mosaic.png"
edges_path_mozaic = "./../output/edges_mozaic.png"
output_prediction_path = "./../output/features.txt"

ImageList = import_mozaic(input_path)
SegementedImageList = segment_images(ImageList)
save_image_list_as_mozaic(SegementedImageList, output_path_mozaic)

#CannyEdgesImageList = get_cedges(SegementedImageList)
#save_image_list_as_mozaic(CannyEdgesImageList, edges_path_mozaic)
FeatureList = get_features(SegementedImageList)

DataSet = build_dataset(FeatureList)

DataSetShuf = np.copy(DataSet)
np.random.shuffle(DataSetShuf)
print(DataSet)
print(DataSetShuf)

X = DataSetShuf[:, 1:3]
Y = DataSetShuf[:, 0]

xx = DataSet[:, 1:3]
yy = DataSet[:, 0]

classifier = svm.SVC(kernel = 'linear', gamma='auto')
classifier.fit(X, Y)


with open(output_prediction_path, 'w') as of:
  for rno in range(40):
    features = xx[rno].reshape(1, -1)
    prediction = classifier.predict(features)
    feat1 = xx[rno][0]
    feat2 = xx[rno][1]
    if(prediction == 0):
      of.write("%.3f, %.3f, 1,"%  (feat1, feat2))
    else:
      of.write("%.3f, %.3f, 2,"%  (feat1, feat2))
    of.write("\n")
