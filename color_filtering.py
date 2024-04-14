import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread( "image15.png" )
down_width = 1024
down_height = 1024
down_points = ( down_width, down_height )
img = cv.resize( img, down_points, interpolation = cv.INTER_LINEAR )

img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#hsv_blur = cv.blur(hsv, (5,5))

#cv.imshow('blurred', hsv_blur) 

l_bound1_contour = np.array([0, 0, 200])
u_bound1_contour = np.array([180, 255, 255])
    
mask_contour = cv.inRange( hsv, l_bound1_contour, u_bound1_contour )    
# The black region in the mask has the value of 0,
# so when multiplied with original image removes all non-blue regions
result = cv.bitwise_and( img, img, mask = mask_contour )

cv.imshow('original', img)
cv.imshow('mask', mask_contour)
cv.imshow('result', result)

cv.imwrite( 'masks/mask8.png', mask_contour )
    
cv.waitKey(0)
cv.destroyAllWindows()

# Might not be accurate though ..
# color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
#               'white': [[180, 18, 255], [0, 0, 231]],
#               'red1': [[180, 255, 255], [159, 50, 70]],
#               'red2': [[9, 255, 255], [0, 50, 70]],
#               'green': [[89, 255, 255], [36, 50, 70]],
#               'blue': [[128, 255, 255], [90, 50, 70]],
#               'yellow': [[35, 255, 255], [25, 50, 70]],
#               'purple': [[158, 255, 255], [129, 50, 70]],
#               'orange': [[24, 255, 255], [10, 50, 70]],
#               'gray': [[180, 18, 230], [0, 0, 40]]}