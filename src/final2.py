import cv2
import imutils
images = []
mountains = ['Mountain_2','Mountain_1','Mountain_3','Mountain_5','Mountain_4','Mountain_6','Mountain_7']
# mountains = ['Mountain_5','Mountain_2','Mountain_1','Mountain_4','Mountain_3','Mountain_7','Mountain_6']
offices = ['office-00','office-01','office-02','office-03']
towers = ['Tower_2','Tower_1']
yards = ['yard-04','yard-05','yard-03','yard-02','yard-01','yard-06','yard-07','yard-08']
imlist = yards
for i in imlist:
    images.append(cv2.imread('../images/Input/{}.jpg'.format(i)))
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
cv2.imwrite('../images/results/res.jpg',stitched)