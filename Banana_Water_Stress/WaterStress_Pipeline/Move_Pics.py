import os
import shutil
from PIL import Image
import cv2
import png
import pandas as pd
from skimage import io
import numpy as np

# Move and crop images
task_path = r'D:\Water Stress\Experiment 601 New Water Stress'
save_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\512_512_pics\Depth'
img_crop = (384, 108, 896, 620)  # Crop sizes

for task in os.listdir(task_path):
    for cur_file in os.listdir(task_path + '/' + task):
        try:
            if len(str(os.listdir(task_path + '/' + task + '/' + cur_file))) < 13:
                if int(task[5:]) <= 814:
                    image = Image.open(
                        task_path + '/' + task + '/' + cur_file + '/Task_' + cur_file[5:7].upper() + cur_file[
                            8].upper() + '/3D_Channel_RealSense_Depth.PNG')
                else:
                    image = Image.open(
                        task_path + '/' + task + '/' + cur_file + '/Frame_' + cur_file[5:7].upper() + cur_file[
                            8].upper() + '/3D_Channel_RealSense_Depth.PNG')

                image = image.crop(img_crop)
                image.save(save_path + '/' + task + '_' + cur_file[5:].upper() + '.PNG', 'PNG')
            else:
                if int(task[5:]) <= 814:
                    image = Image.open(task_path + '/' + task + '/' + cur_file + '/Task_' + cur_file[
                                                                                            5:].upper() + '/3D_Channel_RealSense_Depth.PNG')
                else:
                    image = Image.open(task_path + '/' + task + '/' + cur_file + '/Frame_' + cur_file[
                                                                                             5:].upper() + '/3D_Channel_RealSense_Depth.PNG')

                image = image.crop(img_crop)
                image.save(save_path + '/' + task + '_' + cur_file[5:].upper() + '.PNG', 'PNG')
        except:
            print(cur_file)


# ---------------------------------- Move Annotations - drop out very small leaves ----------------------------------- #

task_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Segmentation_try\leafsegmentor\inference_output\512_leaf_segmentor_inference'
save_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Segmentation_try\leafsegmentor\inference_output\512_leaf_segmentor_clean_inference'

for task in os.listdir(task_path):
    try:
        if (str(task).endswith('t') and str(task)[-5:].startswith('0')) or str(task).lower().endswith('eg'):
            shutil.copy(task_path + '/' + task, save_path + '/' + task)
    except:
        print(task)

# ---------------------------------------- Arrange annotations in folders -------------------------------------------- #

from_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Segmentation_try\leafsegmentor\inference_output\512_leaf_segmentor_inference\512_leaf_segmentor_clean_inference'
dest_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves\New_Segmentor_Annotations'

# Create folders one for each plant
for task in os.listdir(from_path):
    if task.lower().endswith('.jpeg'):
        folder_name = task[:-5]
        os.makedirs(dest_path + '/' + folder_name, exist_ok=True)

# Move annotations of the leaves to the folder of the right plant
for anno in os.listdir(from_path):
    if anno.lower().endswith('.txt'):
        # leaves
        # leaf_name = anno[:-6]
        # task_num = leaf_name[:6]
        # plant_num = leaf_name[8:12]
        # leaf_num = leaf_name[-4:]
        #
        # folder_name = task_num + plant_num
        # leaf_name = task_num + plant_num + leaf_num
        #
        # src_dir = from_path + '/' + anno
        # dst_dir = dest_path + '/' + folder_name + '/' + leaf_name + '.txt'
        # shutil.copy(src_dir, dst_dir)

        # Use this one
        anno_num = anno[15:17]
        folder_name = anno[:13]
        src_dir = from_path + '/' + anno
        dst_dir = dest_path + '/' + folder_name + '/' + folder_name + '_' + anno_num + '.txt'
        shutil.copy(src_dir, dst_dir)




