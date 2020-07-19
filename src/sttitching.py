import numpy as np
import cv2
import copy
import cmath


def interpolation(img):
    cop_im = copy.deepcopy(img)
    for i in range(0,img.shape[0]):
        for j in range(0,img.shape[1]):
            for k in range(0,img.shape[2]):
                if (img[i,j,k] == 1):
                    find = False
                    radius = 1
                    while(not find):
                        try:
                            if (img[i+radius,j,k] != 1):
                                cop_im[i,j] = img[i+radius,j]
                                find = True
                        except:
                            pass
                        try:
                            if img[i-radius,j,k] != 1:
                                cop_im[i,j] = img[i-radius,j]
                                find = True
                        except:
                            pass
                        try:
                            if img[i,j+radius,k] != 1:
                                cop_im[i,j] = img[i,j+radius]
                                find = True
                        except:
                            pass
                        try:
                            if img[i,j-radius,k] != 1:
                                cop_im[i,j] = img[i,j-radius]
                                find = True
                        except:
                            pass
                        if radius == 2:
                            find = True
                        radius += 1

    return cop_im

def median_filter(img,sizeWindow):
    res = copy.deepcopy(img)
    paddingW2cntri = int(sizeWindow[0]/2)
    paddingW2cntrj = int(sizeWindow[1]/2)
    for i in range(paddingW2cntri,img.shape[0]-paddingW2cntri):
        for j in range(paddingW2cntrj,img.shape[1]-paddingW2cntrj):
            for k in range(3):
                if(img[i,j,k]==1 ):
                    window = np.sort(np.asarray(img[i-paddingW2cntri:i+paddingW2cntri+1,j-paddingW2cntrj:j+paddingW2cntrj+1,k]).flatten())
                    res[i,j,k] = window[int(sizeWindow[0]*sizeWindow[1]/2)]
    return res


def max_filter(img,sizeWindow):
    res = copy.deepcopy(img)
    paddingW2cntri = int(sizeWindow[0]/2)
    paddingW2cntrj = int(sizeWindow[1]/2)
    for i in range(paddingW2cntri,img.shape[0]-paddingW2cntri):
        for j in range(paddingW2cntrj,img.shape[1]-paddingW2cntrj):
            for k in range(3):
                if(img[i,j,k]==1 ):
                    window = np.sort(np.asarray(img[i-paddingW2cntri:i+paddingW2cntri+1,j-paddingW2cntrj:j+paddingW2cntrj+1,k]).flatten())
                    res[i,j,k] = window[int(sizeWindow[0]*sizeWindow[1])-1]
    return res


def stitchIm(kps1,kps2,rgb_im1,rgb_im2,h):
    size_im1 = rgb_im1.shape

    size_im2 = rgb_im2.shape
    S = 4
    res_im = np.ones((int(size_im1[1]+S*size_im2[1]),int(size_im1[0]+S*size_im2[0]),3),dtype=np.uint8)
    print(res_im.shape)
    res_im[int(S/2)*rgb_im2.shape[0]:int(S/2)*rgb_im2.shape[0] + rgb_im1.shape[0]
    , int(S/2)*rgb_im2.shape[1]:int(S/2)*rgb_im2.shape[1] + rgb_im1.shape[1]] = rgb_im1

    for i in range(0,size_im2[0]):
        for j in range(0,size_im2[1]):
            xnew = int( ((h[0,0]*j)+(h[0,1]*i)+h[0,2]) /
                        ((h[2,0]*j)+(h[2,1]*i)+h[2,2]))
            ynew = int( ((h[1,0]*j)+(h[1,1]*i)+h[1,2])
                        /((h[2,0]*j)+(h[2,1]*i)+h[2,2]))
            # xnew = int( ((h[0,0]*j)+(h[0,1]*i)+h[0,2]) )
            # ynew = int( ((h[1,0]*j)+(h[1,1]*i)+h[1,2]))
            xnew += int(S/2)*size_im2[1]
            ynew += int(S/2)*size_im2[0]
            for k in range(3):
                res_im[ynew,xnew,k] = rgb_im2[i,j,k]
    xmax = np.argmax(res_im, axis=1)
    minRow = 0
    for i in range(xmax.shape[0]):
        if (xmax[i,0]!=0 or xmax[i,1]!=0 or xmax[i,2]!=0 ):
            minRow = i
            print(minRow)
            break;
    maxRow = 0
    for i in range(xmax.shape[0]-1,0,-1):
        if (xmax[i,0]!=0 or xmax[i,1]!=0 or xmax[i,2]!=0 ):
            maxRow = i
            print(maxRow)
            break;

    ymax = np.argmax(res_im, axis=0)
    minCol = 0
    for i in range(ymax.shape[0]):
        if (ymax[i,0]!=0 or ymax[i,1]!=0 or ymax[i,2]!=0 ):
            minCol = i
            print(minCol)
            break;
    maxCol = 0
    for i in range(ymax.shape[0]-1,0,-1):
        if (ymax[i,0]!=0 or ymax[i,1]!=0 or ymax[i,2]!=0 ):
            maxCol = i
            print(maxCol)
            break;
    res_im = res_im[minRow:maxRow,minCol:maxCol,:]

    # fp = res_im[:,:,2]
    # Fuv = np.fft.fft2(fp)
    # FuvA,FuvP = toPolar(Fuv)
    # FuvA = np.fft.fftshift(FuvA)
    # FuvA += 1
    # showRES = np.log10(FuvA)
    # min = np.min(showRES)
    # max = np.max(showRES)
    # showRES = normalizeimg(showRES,min,max)

    # res_im = max_filter(res_im,(3,3))
    # print("median1")
    cv2.imwrite("res.jpg",res_im)
    # cv2.imwrite("freqRes.jpg",showRES)
    return res_im

