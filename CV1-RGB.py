# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:16:33 2023

@author: 86158
"""

import cv2
import math
import numpy as np

def DarkChannel(img,size): #暗通道
    b,g,r = cv2.split(img)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(img,dark): #大气光照
    [h,w] = img.shape[:2]
    img_size = h*w
    numpx = int(max(math.floor(img_size/1000),1))
    darkvec = dark.reshape(img_size)
    imvec = img.reshape(img_size,3)

    indices = darkvec.argsort()
    indices = indices[img_size-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(img,A,sz):
    omega = 0.95
    img3 = np.empty(img.shape,img.dtype)

    for ind in range(0,3):
        img3[:,:,ind] = img[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(img3,sz)
    return transmission

def GuidedFilter(img,p,r,eps): #引导滤波
    mean_I = cv2.boxFilter(img,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(img*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(img*img,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    return mean_a*img + mean_b

def TransmissionRefine(img,et):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001

    return GuidedFilter(gray,et,r,eps)

def Recover(img,t,A,tx = 0.1):
    res = np.empty(img.shape,img.dtype)
    t = cv2.max(t,tx)

    for ind in range(0,3):
        res[:,:,ind] = (img[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res


def run(fn):
    print("Processing image: %s"%(fn))
    start = time.time()
    
    src = cv2.imread(fn)
    I = src.astype('float64')/255
    
    cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
    cv2.imshow('preview', I)

    dark = DarkChannel(I,7)
    A = AtmLight(I,dark)
    te = TransmissionEstimate(I,A,15)
    t = TransmissionRefine(src,te)
    J = Recover(I,t,A,0.1)
    
    print(time.time() - start);
    cv2.namedWindow("defog", cv2.WINDOW_NORMAL)
    cv2.imshow('defog',J)
    cv2.waitKey()

if __name__ == '__main__':
    import time

    for i in range(1,10):
        
        fn = './' + str(i) + '.jpg'

        run(fn)

    
    