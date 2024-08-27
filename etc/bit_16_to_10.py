from tqdm import tqdm
import os
import glob

path='/mnt/sdb/4월촬영본_16bit_tiff'
for im in tqdm(sorted(glob.glob(path + '/*/*/*.tiff'))):
    print(im)
    os.system(f'convert -depth 12 {im} {im}')
    
#convert : imagemagic 설치해 놓은 것
#이미지 확인 command : identify -verbose {이미지}.tiff
