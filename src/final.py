import numpy as np
import cv2

from src.sttitching import *


def findBesMatch(imgs):
    kps   = []
    deses = []
    for i,img in enumerate(imgs):
        surf     = cv2.xfeatures2d.SURF_create(400)
        img1 = img1.astype('uint8')
        kp, des = surf.detectAndCompute(img1,None)
        kps.append(kp)
        deses.append(des)
    n = len(kps)
    bestNum = 0
    best = (0,1)
    for i in range(n-1):
        for j in range(i+1,n-1):
            # BFMatcher with default params
            bf      = cv2.BFMatcher()
            matches = bf.knnMatch(deses[i],deses[j],k=2)
            good_matches = []
            for m,n in matches:
                if m.distance < 0.25*n.distance:
                    good_matches.append([m])
            if len(good_matches) > bestNum:
                best = (i,j)
                bestNum = len(good_matches)

    return imgs[i],imgs[j],kps[i],kps[j]


def stitch(img1,img2):
    surf     = cv2.xfeatures2d.SURF_create(400)
    img1 = img1.astype('uint8')
    kp1, des1 = surf.detectAndCompute(img1,None)
    kp2, des2 = surf.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf      = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.25*n.distance:
            good_matches.append([m])
    # good_matches = matches
    # Select good matched keypoints
    print(len(good_matches))
    matched_kpts1   = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    matched_kpts2   = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    # Compute homography
    # H, status    = cv2.findHomography(matched_kpts2, matched_kpts1, cv2.RANSAC,10)
    H = cv2.getAffineTransform(matched_kpts2[:3], matched_kpts1[:3])
    print(H)

    # Warp image
    # res_im = cv2.warpPerspective(img2,M=H,dsize=(img1.shape[1]+img2.shape[1], img1.shape[0]))
    res_im = stitchIm(matched_kpts1,matched_kpts2,img1,img2,H)

    return res_im





def main():
    mountains = ['Mountain_2','Mountain_1','Mountain_3','Mountain_5','Mountain_4','Mountain_6','Mountain_7']
    # mountains = ['Mountain_5','Mountain_2','Mountain_1','Mountain_4','Mountain_3','Mountain_7','Mountain_6']
    offices = ['office-00','office-01','office-02','office-03']
    towers = ['Tower_2','Tower_1']
    yards = ['yard-04','yard-05','yard-03','yard-02','yard-01','yard-06','yard-07','yard-08']
    imlist = mountains
    ref_im = cv2.imread('../images/Input/{}.jpg'.format(imlist[0]))
    for i,imName in enumerate(imlist[1:]):
        scene_im = cv2.imread('../images/Input/{}.jpg'.format(imName))
        ref_im = stitch(ref_im,scene_im)
        print("here")
        cv2.imwrite('../images/results/result{}.jpg'.format(imName), ref_im)


    # ref_im = median_filter(ref_im,(3,3))
    # print("median1")
    # ref_im = median_filter(ref_im,(3,3))
    # print("median2")
    # ref_im = median_filter(ref_im,(3,3))
    # print("median3")

    cv2.imwrite('../images/results/result{}.jpg'.format(imName), ref_im)
    # cv2.imwrite('../images/results/fft-2-{}.jpg'.format(imName), showRES)

main()
