#6월 촬영본 중 4k
import os
import cv2
import datetime
import argparse
from tqdm import tqdm 

def extract_number(directory):
    parts = directory.split('C')
    if len(parts) > 1:
        first_part =parts[0] #parts[-1]
        second_part = parts[-1]
        try:
            return int(second_part)
        except ValueError:
            return float('inf')  # Return infinity for non-numeric parts (optional)
    return float('inf')  # Return infinity if no underscore is found (optional)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=str)
    parser.add_argument('output', type=str)
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.inputs)
    # cap = cv2.VideoCapture("/mnt/nas3/8K 촬영본/8k_org/A001_0527N0.RDM/A001_C001_0527Q9.RDC/A001_C001_0527Q9_001.mov")
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("length",length)
    print("fps",fps)


    for i, m in tqdm(enumerate(range(0,int(length/fps),10))):
        time1=datetime.datetime.now()
    
        os.makedirs(f'{args.output}/{str(i).zfill(4)}', exist_ok=True)
        
        starttime = str(datetime.timedelta(seconds=m))
        if len(starttime) == 7:
            starttime = '0'+starttime
        endtime = str(datetime.timedelta(seconds=m+2))
        if len(endtime) == 7:
            endtime = '0'+endtime
        print(i)
        print("starttime",starttime)
        print("endtime",endtime)
        os.system(f'ffmpeg -ss {starttime} -to {endtime} -i "{args.inputs}" -vf "fps=10,scale=3840:2160" -pix_fmt rgb48 "{args.output}/{str(i).zfill(4)}/%08d.tiff"') #crop까지 
        time2=datetime.datetime.now()
        print(time2-time1)
