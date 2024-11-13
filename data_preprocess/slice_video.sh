#!/bin/bash

# 각 타임 스탬프에 맞는 영상 추출 명령어
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:00:00 -to 00:00:03 -c copy output1.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:00:49 -to 00:00:52 -c copy output2.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:01:05 -to 00:01:10 -c copy output3.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:01:27 -to 00:01:34 -c copy output4.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:02:07 -to 00:02:11 -c copy output5.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:02:50 -to 00:02:54 -c copy output6.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:03:01 -to 00:03:05 -c copy output7.wav
ffmpeg -i /mnt/sdc/dragon_wav_edit/ep159.wav -ss 00:03:27 -to 00:03:32 -c copy output8.wav


# 합치기 위해 파일 리스트 작성
echo "file 'output1.wav'" >> file_list.txt
echo "file 'output2.wav'" >> file_list.txt
echo "file 'output3.wav'" >> file_list.txt
echo "file 'output4.wav'" >> file_list.txt
echo "file 'output5.wav'" >> file_list.txt
echo "file 'output6.wav'" >> file_list.txt
echo "file 'output7.wav'" >> file_list.txt
echo "file 'output8.wav'" >> file_list.txt

# 리스트된 영상들을 하나의 파일로 합치기
ffmpeg -f concat -safe 0 -i file_list.txt -c copy final_output_159.wav

# ffmpeg -i final_output.mpg -vf "scale=out_color_matrix=bt470m,setsar=128/135,setdar=4/3"  -c:v prores_ks -profile:v 4 -qscale:v 0 final_output.mpg

# 임시 파일 정리
rm file_list.txt
