import numpy as np
import cv2

from src.sttitching import *


def initialfindBesMatch(imgs):
    kps   = []
    deses = []
    for i,img in enumerate(imgs):
        surf     = cv2.xfeatures2d.SURF_create(400)
        img = img.astype('uint8')
        kp, des = surf.detectAndCompute(img,None)
        kps.append(kp)
        deses.append(des)
    bestNum = 000
    best = (0,1)
    bestGoodMatches = []
    for i in range(len(kps)):
        for j in range(i+1,len(kps)):
            # BFMatcher with default params
            bf      = cv2.BFMatcher()
            matches = bf.knnMatch(deses[i],deses[j],k=2)
            good_matches = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good_matches.append([m])
            if len(good_matches) > bestNum:
                best            = (i,j)
                bestNum         = len(good_matches)
                bestGoodMatches = good_matches


    print(bestNum)
    return best[0],best[1],bestGoodMatches,kps,deses


def findBestMatch(img,kps,deses):
    surf     = cv2.xfeatures2d.SURF_create(400)
    img = img.astype('uint8')
    kp1, des1 = surf.detectAndCompute(img,None)
    bestNum = 1000
    best = (0,1)
    bestGoodMatches = []
    for i in range(len(kps)):
        # BFMatcher with default params
        bf      = cv2.BFMatcher()
        matches = bf.knnMatch(des1,deses[i],k=2)
        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append([m])
        if len(good_matches) > bestNum:
            best            = i
            bestNum         = len(good_matches)
            bestGoodMatches = good_matches

    kp2 = kps[best]
    matched_kpts1   = np.float32([kp1[m[0].queryIdx].pt for m in bestGoodMatches])
    matched_kpts2   = np.float32([kp2[m[0].trainIdx].pt for m in bestGoodMatches])
    print(bestNum)
    return best,matched_kpts1,matched_kpts2

def stitch(imgs):
    # surf     = cv2.xfeatures2d.SURF_create(400)
    # img1 = img1.astype('uint8')
    # kp1, des1 = surf.detectAndCompute(img1,None)
    # kp2, des2 = surf.detectAndCompute(img2,None)
    #
    # # BFMatcher with default params
    # bf      = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2,k=2)
    # good_matches = []
    # for m,n in matches:
    #     if m.distance < 0.25*n.distance:
    #         good_matches.append([m])
    # # good_matches = matches
    # # Select good matched keypoints
    # print(len(good_matches))
    # matched_kpts1   = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
    # matched_kpts2   = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])
    # Compute homography
    # H, status    = cv2.findHomography(matched_kpts2, matched_kpts1, cv2.RANSAC,10)
    # H = cv2.getAffineTransform(matched_kpts2[:3], matched_kpts1[:3])
    c=0
    img1Idx,img2Idx,bestGoodMatches,kps,deses = initialfindBesMatch(imgs)
    kp1 = kps[img1Idx]
    kp2 = kps[img2Idx]
    matched_kpts1   = np.float32([kp1[m[0].queryIdx].pt for m in bestGoodMatches])
    matched_kpts2   = np.float32([kp2[m[0].trainIdx].pt for m in bestGoodMatches])
    H = cv2.getAffineTransform(matched_kpts2[:3], matched_kpts1[:3])
    img2 = imgs.pop(img2Idx)
    img1 = imgs.pop(img1Idx)
    kps.pop(img2Idx)
    kps.pop(img1Idx)
    deses.pop(img2Idx)
    deses.pop(img1Idx)
    print(H)
    res_im = stitchIm(img1,img2,H)
    cv2.imwrite('../images/results/{}.jpg'.format(c),res_im)
    while(imgs):
        c+=1
        img1Idx,matched_kpts1,matched_kpts2 = findBestMatch(res_im,kps,deses)
        H = cv2.getAffineTransform(matched_kpts2[:3], matched_kpts1[:3])
        img1 = imgs.pop(img1Idx)
        kps.pop(img1Idx)
        deses.pop(img1Idx)
        print(H)
        res_im = stitchIm(res_im,img1,H)
        cv2.imwrite('../images/results/{}.jpg'.format(c),res_im)

    # Warp image
    # res_im = cv2.warpPerspective(img2,M=H,dsize=(img1.shape[1]+img2.shape[1], img1.shape[0]))
    # res_im = stitchIm(matched_kpts1,matched_kpts2,img1,img2,H)

    return res_im





def main():
    mountains = ['Mountain_2','Mountain_1','Mountain_3','Mountain_5','Mountain_4','Mountain_6','Mountain_7']
    # mountains = ['Mountain_5','Mountain_2','Mountain_1','Mountain_4','Mountain_3','Mountain_7','Mountain_6']
    offices = ['office-00','office-01','office-02','office-03']
    towers = ['Tower_2','Tower_1']
    yards = ['yard-04','yard-05','yard-03','yard-02','yard-01','yard-06','yard-07','yard-08']
    imlist = mountains
    # ref_im = cv2.imread('../images/Input/{}.jpg'.format(imlist[0]))
    # for i,imName in enumerate(imlist[1:]):
    #     scene_im = cv2.imread('../images/Input/{}.jpg'.format(imName))
    #     ref_im = stitch(ref_im,scene_im)
    #     print("here")
    #     cv2.imwrite('../images/results/result{}.jpg'.format(imName), ref_im)
    imgs = []
    for imName in imlist:
        imgs.append(cv2.imread('../images/Input/{}.jpg'.format(imName)))
    ref_im = stitch(imgs)

    # ref_im = median_filter(ref_im,(3,3))
    # print("median1")
    # ref_im = median_filter(ref_im,(3,3))
    # print("median2")
    # ref_im = median_filter(ref_im,(3,3))
    # print("median3")

    cv2.imwrite('../images/results/result{}.jpg'.format(imName), ref_im)
    # cv2.imwrite('../images/results/fft-2-{}.jpg'.format(imName), showRES)

main()
