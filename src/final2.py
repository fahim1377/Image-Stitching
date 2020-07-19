import cv2
import imutils
from skimage.metrics import structural_similarity as compare_ssim
import matplotlib.pyplot as plt


images = []
mountains = ['Mountain_2','Mountain_1','Mountain_3','Mountain_5','Mountain_4','Mountain_6','Mountain_7']
# mountains = ['Mountain_5','Mountain_2','Mountain_1','Mountain_4','Mountain_3','Mountain_7','Mountain_6']
# mountains = ['Mountain_2','Mountain_3','Mountain_4','Mountain_5','Mountain_6','Mountain_7','Mountain_1']
offices = ['office-00','office-01','office-02','office-03']
towers = ['Tower_2','Tower_1']
yards = ['yard-04','yard-05','yard-03','yard-02','yard-01','yard-06','yard-07','yard-08']
imlist = yards
Ref_im = 'yard_Ref'
for i in imlist:
    images.append(cv2.imread('../images/Input/{}.jpg'.format(i)))
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)
cv2.imwrite('../images/results/yard.jpg',stitched)

reference_im = cv2.imread('../images/Reference/{}.jpg'.format(Ref_im))
ref_im = cv2.resize(stitched,(reference_im.shape[1],reference_im.shape[0]))
score , diff = compare_ssim(reference_im,ref_im,multichannel=True,full=True)


data = [[" ",score]]

#display mmse table
types = ["ssim","yard"]
col_labels = tuple(types)
fig, ax = plt.subplots(dpi=300, figsize=(5,5))
ax.axis('off')
ax.table(cellText=[data], colLabels=col_labels, loc='center')
fig.savefig('../images/results/resTable.png')

