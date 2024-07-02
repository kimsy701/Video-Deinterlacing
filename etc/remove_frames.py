import glob
import os
import shutil
from icecream import ic
from tqdm import tqdm

def remove_files_by_index_range(base_dir, index_ranges):
    for i, img in enumerate(sorted(glob.glob(base_dir + '*/*'))):
        img_dir = os.path.dirname(img) 
        #print(img_dir)
        #print(i , img) #16684 /mnt/sdd/2D_Ani/날씨의아이/PNG_dataset/scene_00000235/날씨의아이_1020411.png    #/media/user/My Book/inshorts/스즈메의문단속/PNG_dataset/scene_00000000/Thumbs.db
        #print(img.split('/')[-1].split('_')[-1].split('.')[0]    ,1)

        actual_num = int(img.split('/')[-1].split('_')[-1].split('.')[0])
        for start, end in index_ranges:
            if start <= actual_num <= end:
                print(f' find:{start}~{end} |  {actual_num}')
                ic(i, img, "REMOVED")
                #os.system(f'rm "{img_dir}/{name}{str(i).zfill(6)}.png"')
                
                os.remove(img)
                #Check if the directory is empty after removing the file

                if not os.listdir(img_dir):
                    print(f"Removing folder: {img_dir}")
                    shutil.rmtree(img_dir)


def remove_folder_15frames(dir):
    for flds in os.listdir(dir):
        each_fld = os.path.join(dir,flds)
        numfiles = os.listdir(each_fld)
        if len(numfiles) < 15:
            for file in numfiles:
                file_path = os.path.join(each_fld, file)
                print(f' REMOVING: {file_path}')
                os.remove(file_path)
                if not os.listdir(each_fld):
                    print(f'Removing folder: {each_fld}')
                    shutil.rmtree(each_fld)
        else:
            print(f"scene folder {flds} contains more than 20 files. Not removed.")


def remove_frames_more_than_200(base_path):
    #for vid in tqdm(sorted(os.listdir(base_path))):
        for scene in tqdm(sorted(os.listdir(base_path))):
            if '.DS_Store' not in scene:
                scene_fld_path = os.path.join(base_path, scene)
                # move 50 frames to new fld files
                img_list = [img for img in sorted(os.listdir(scene_fld_path))]
                #print(img_list)

                if len(img_list) > 200:
                    print(scene_fld_path, len(img_list))
                    for img in img_list[50:]:
                        
                        #print(scene_fld_path+'/'+img)
                        os.remove(scene_fld_path+'/'+img)

def remove_empty_folders(root_folder):
    # Walk through the directory tree from bottom up
    for folder, _, _ in os.walk(root_folder, topdown=False):
        try:
            os.rmdir(folder)  # Try to remove the directory
            print(f"Removed empty folder: {folder}")
        except OSError as e:
            print(f"Error: {e}")

#remove_empty_folders('/mnt/sdd/2D_Ani/날씨의아이/PNG_dataset')


def main():

    #remove indexed frames
    #dir = '/media/user/My Book/inshorts/스즈메의문단속/PNG_dataset/'
    #dir = '/mnt/sde/2D_Ani/스즈메의문단속/PNG_dataset/'
    #dir = '/mnt/sdd/2D_Ani/날씨의아이/PNG_dataset/'
    dir = '/mnt/sda/2D_Ani/너의이름은/PNG_dataset/'
       
    #index_range = [(1000000, 1001331)]    #원래 : [(0, 1331)] #날씨의 아이 index
    #index_range = [(1000000, 1000743)]    #원래 : [(0, 743)] #스즈메의문단속 index
    index_range = [(1000000, 1000475)]    #원래 : [(0, 475)] #너의이름은 index #[(1000000, 10000475)]  아님..
    remove_files_by_index_range(dir, index_range)
    
    #index_range = [(1151783, 1161447)]    #원래 : [(151783, 161447)] #날씨의 아이 index
    #index_range = [(1018470, 1018793)]    #원래 : [(18470, 18793)] #스즈메의문단속 index
    index_range = [(1145638, 1153341)]    #원래 : [(145638, 153341)] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    #index_range = [(1167790, 1174672)]    #원래 : [(167790 , 174672 )] #스즈메의문단속 index
    index_range = [(1001248, 1001367)]    #원래 : [(1248, 1367)] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1002679, 1002867)]    #원래 : [(2679, 2867)] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1004810, 1004944)]    #원래 : [(4810, 4944)] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1063232, 1063387)]    #원래 : [(63232, 63387 )] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1091525, 1091643 )]    #원래 : [(91525, 91643 )] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1132891, 1132991 )]    #원래 : [(132891, 132991 )] #너의이름은 index
    remove_files_by_index_range(dir, index_range)
    
    index_range = [(1133448, 1133590 )]    #원래 : [(133448 , 133590  )] #너의이름은 index
    remove_files_by_index_range(dir, index_range)


    #remove less then 15, more than 200 frame
    #dir = '/media/user/My Book/inshorts/스즈메의문단속/PNG_dataset'
    #dir='/mnt/sdd/2D_Ani/날씨의아이/PNG_dataset'
    #remove_folder_15frames(dir) #수행1: 수행한 후 주석처리

    #remove_frames_more_than_200(dir) #수행2: 주석처리 제거 후 수행
    
    #remove_empty_folders(dir)

# if __name__ == "__main__":
#     main()
