import numpy as np
import cv2
cap = cv2.VideoCapture('fix.mp4') #Open video file
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True) #Create the background substractor
kernelOp = np.ones((3,3),np.uint8)
kernelCl = np.ones((11,11),np.uint8)
while(cap.isOpened()):
    ret, frame = cap.read() #read a frame
    fgmask = fgbg.apply(frame) #Use the substractor
    try:
        ret,imBin= cv2.threshold(fgmask,200,155,cv2.THRESH_BINARY)
        #Opening (erode->dilate) 
        mask = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        #Closing (dilate -> erode) 
        mask = cv2.morphologyEx(mask , cv2.MORPH_CLOSE, kernelCl)
    except:
        #if there are no more frames to show...
        print('EOF')
        break
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(frame, cnt, -1, (0,155,0), 3, 8)
        cv2.imshow('Frame',frame)
    #Abort and exit with 'Q' or ESC
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows