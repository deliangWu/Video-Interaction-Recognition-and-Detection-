import numpy as np
import cv2 as cv

img = cv.imread('D:/PIC/1.jpg')
resizeRatio = 0.1
img = cv.resize(img,(int(img.shape[1] * resizeRatio), int(img.shape[0] * resizeRatio)), interpolation= cv.INTER_AREA)
filter_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
filter_kernel_t = filter_kernel.transpose(1,0)
print(filter_kernel)
print(filter_kernel_t)
img_conv2d = cv.filter2D(img, ddepth=-1, kernel=filter_kernel)
img_conv2d_t = cv.filter2D(img_conv2d, ddepth=-1, kernel=filter_kernel)
img_show = np.append(img_conv2d_t,img_conv2d,0)

cv.namedWindow('IMAGE')
while True:
    cv.imshow('IMAGE',img_show)
    if cv.waitKey(30) == 27:
        break


    
    