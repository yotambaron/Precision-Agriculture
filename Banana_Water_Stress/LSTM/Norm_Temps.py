from cv2 import cv2
import pandas as pd
import numpy as np
from PIL import Image


# Set save paths
save_path_rgb = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\RGB_Masked_m1'
save_path_thermal = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Thermal_Masked_m1'
save_path_depth = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Depth_Masked_m1'

# Set image paths
image_path_rgb = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\RGB_Masked_0'
image_path_thermal = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Thermal_masked_0'
image_path_depth = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Depth_Masked_0'

# Load temperatures to normalize
temps_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Other\Temperature analysis'
temps = pd.read_csv(temps_path + '/thermal_interpolated_values_exp_601.csv')

# Loop through every plant in all days
for ind, plant in enumerate(temps['tasknd']):

    # Load current rgb, thermal and depth images
    # rgb_image = cv2.imread(image_path_rgb + '/' + plant + '.JPEG', -1)
    thermal_image = cv2.imread(image_path_thermal + '/' + plant + '.PNG', -1)
    # depth_image = cv2.imread(image_path_depth + '/' + plant + '.PNG', -1)

    # Norm thermal images and mask all with (-1) instead of 0
    temp_curr = temps.at[ind, 'temperature'] * 100
    thermal_image_norm = np.where(thermal_image == 0, 0, np.int8(thermal_image - temp_curr))
    # rgb_image_norm = np.where(rgb_image == 0, -1, rgb_image)
    # depth_image_norm = np.where(depth_image == 0, -1, depth_image)

    # Save all images
    # cv2.imwrite(save_path_rgb + '/' + plant + '.JPEG', rgb_image_norm)
    cv2.imwrite(save_path_thermal + '/' + plant + '.PNG', thermal_image_norm)
    # cv2.imwrite(save_path_depth + '/' + plant + '.PNG', depth_image_norm)

    print(ind, plant)




