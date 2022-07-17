import cv2
import numpy as np


cap = cv2.VideoCapture(0)
imgtarget = cv2.imread('Marker.png')
myimg = cv2.VideoCapture('lofi.jpg')

detection = False
framecounter = 0

success, imgart = myimg.read()
hT,wT,cT = imgtarget.shape
imgart = cv2.resize(imgart,(wT,hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgtarget,None)

while (True):
    sucess, imgwebcam = cap.read()
    imgaug = imgwebcam.copy()
    kp2, des2 = orb.detectAndCompute(imgwebcam,None)
    #imgwebcam = cv2.drawKeypoints(imgwebcam,kp2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance<0.75 *n.distance:
            good.append(m)
    print(len(good))
    imgfeatures = cv2.drawMatches(imgtarget,kp1,imgwebcam,kp2,good,None,flags=2)

    if len(good) > 50:
        scrpts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dstpts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        matrix, mask = cv2.findHomography(scrpts,dstpts,cv2.RANSAC,5)
        print(matrix)

        pts = np.float32([[0,0],[0,hT],[wT,hT],[wT,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, matrix)
        img2 = cv2.polylines(imgwebcam,[np.int32(dst)], True, (255,0,255),3)
        imgwrap = cv2.warpPerspective(imgart, matrix, (imgwebcam.shape[1],imgwebcam.shape[0]))

        masknew = np.zeros((imgwebcam.shape[0],imgwebcam.shape[1]),np.uint8)
        cv2.fillPoly(masknew,[np.int32(dst)], (255,255,255) )
        maskinv = cv2.bitwise_not(masknew)
        imgaug = cv2.bitwise_and(imgaug, imgaug, mask = maskinv)
        imgaug = cv2.bitwise_or(imgwrap, imgaug)

    cv2.imshow("imgstacked", imgaug)
    cv2.waitKey(1)

    



