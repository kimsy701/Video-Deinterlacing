import cv2
import numpy as np
import os
from PIL import Image
import seaborn as sns

gt_path = "C:/Users/인쇼츠/Desktop/Deinterlacing구현/code/training code/inference_100folders/gt_val_100_re/89_138/im1.png"
# gt_path = "C:\Users\인쇼츠\Desktop\Deinterlacing구현\inference_data\겨울연가\원본\00000941.png"
pred_path = "C:/Users/인쇼츠/Desktop/Deinterlacing구현/code/training code/inference_100folders/QTGMC/out_img2/00000001.png"
dest_path = "C:/Users/인쇼츠/Desktop/Deinterlacing구현/code/training code/inference_100folders/QTGMC/errormap/89_138/im1.png"

gt = Image.open(gt_path)
pred = Image.open(pred_path)

# errormap = np.abs(np.array(pred)*255 - np.array(gt)*255)
errormap = np.abs(np.array(pred) - np.array(gt))
averaged_errormap = np.mean(errormap, axis=2)

# Image.fromarray(errormap).save(dest_path)
cmap = sns.cm.rocket_r
ax =sns.heatmap(averaged_errormap, cmap=cmap, xticklabels=False, yticklabels=False)
fig = ax.get_figure()
fig.savefig(dest_path)
