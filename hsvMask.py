import cv2 
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

img = cv2.imread("input_image.jpg")
# img  = np.flip(img,axis=1)
imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

sensitivity=20
lower_green  = np.array([60 - sensitivity, 100, 100])#(50, 100, 60)#'hsl(%d, %d%%, %d%%)' % (120, 50, 25) #hsla(120, 50%, 25%, 1)
upper_green  = np.array([60 + sensitivity, 255, 255])
mask1 = cv2.inRange(imgHsv, lower_green, upper_green)


# lower_green2 = np.array([72,52,72])
# upper_green2= np.array([97,255,255])
# mask2 = cv2.inRange(imgHsv, lower_green2, upper_green2)
# print(mask2[10])
# cv2.imwrite('/home/salah/Desktop/1003/mask2.jpeg',mask2)

# mask1 = mask1+mask2
# print(mask1[10])
# mask2 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((,2),np.uint8))
maskx = cv2.morphologyEx(mask1, cv2.MORPH_ERODE, np.ones((2,2),np.uint8))
mask2 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((4,4),np.uint8))
# mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))

mask3 = mask1+mask2+maskx
# print('mask2',mask2[10])
# mask3 = cv2.bitwise_not(mask2)

# print(mask3[10])
def myfunc(img,mask):
	green = np.zeros_like(img, np.uint8)
	for i,dim1 in enumerate(mask):
		for j,dim2 in enumerate(dim1):
			if mask[i][j]>0:
				green[i][j]=np.array([255,255,255])
			else:
				green[i][j]=img[i][j]
	return green
res1=myfunc(img,mask3)



cv2.imwrite('output_image.jpeg',res1)
