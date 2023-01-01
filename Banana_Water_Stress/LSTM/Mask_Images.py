from cv2 import cv2
import numpy as np
import os


def switch_x_y(annotation):
    text_list = list(annotation)
    for i, el in enumerate(text_list):
        text_list[i] = (el[1], el[0])
    return text_list


# Set paths
plant_annotations = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Plant\New_Segmentor_Annotations'
leaves_annotations = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
save_path_rgb = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\RGB_Masked'
save_path_thermal = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Thermal_Masked'
save_path_depth = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Depth_Masked'
image_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics'
rgb_path = image_path + '/RGB'
thermal_path = image_path + '/Thermal'
depth_path = image_path + '/Depth'


# Loop through every plant in all days
for ind, anno in enumerate(os.listdir(plant_annotations)):

    # Load rgb, thermal and depth images
    rgb_image = cv2.imread(rgb_path + '/' + anno + '.JPEG', -1)
    thermal_image = cv2.imread(thermal_path + '/' + anno + '.PNG', -1)
    depth_image = cv2.imread(depth_path + '/' + anno + '.PNG', -1)

    # Load plant and leaves annotations of the current plant
    curr_plant_path = plant_annotations + '/' + anno
    curr_leaves_path = leaves_annotations + '/' + anno
    plant_poly_array = []
    leaves_poly_array = []

    try:
        # Save all current plant's annotations in an array as polygons
        for plant in os.listdir(curr_plant_path):
            plant_text = np.loadtxt(curr_plant_path + '/' + plant, delimiter=',')
            plant_text = switch_x_y(plant_text)
            plant_poly_array.append(plant_text)
    except:
        print("No plant annotations: ", anno)

    try:
        # Save all current plant's leaves annotations in an array as polygons
        for leaf in os.listdir(curr_leaves_path):
            if "points" not in leaf and not leaf.endswith('JPEG'):
                leaf_text = np.loadtxt(curr_leaves_path + '/' + leaf, delimiter=',')
                plant_poly_array.append(leaf_text)
    except:
        print("No leaf annotations: ", anno)

    # Create masks for rgb image and for thermal + depth images
    mask_rgb = np.zeros(rgb_image.shape, dtype=np.uint8)
    mask_others = np.zeros(thermal_image.shape, dtype=np.uint8)
    channel_count_rgb = rgb_image.shape[2]
    ignore_mask_color_rgb = (255, ) * channel_count_rgb

    # Loop through all plant's leaves and plant annotations
    for index, poly in enumerate(plant_poly_array):
        poly = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask_rgb, [poly], ignore_mask_color_rgb)   # Fill white the rgb image with the current polygon
        cv2.fillPoly(mask_others, [poly], 255)  # Fill white the thermal and depth images with the current polygon

    # Mask black outside the polygon in all images
    rgb_masked_image = cv2.bitwise_and(rgb_image, mask_rgb)
    thermal_masked_image = cv2.bitwise_and(thermal_image, thermal_image, mask=mask_others)
    depth_masked_image = cv2.bitwise_and(depth_image, depth_image, mask=mask_others)

    # Save all images
    cv2.imwrite(save_path_rgb + '/' + anno + '.JPEG', rgb_masked_image)
    cv2.imwrite(save_path_thermal + '/' + anno + '.PNG', thermal_masked_image)
    cv2.imwrite(save_path_depth + '/' + anno + '.PNG', depth_masked_image)

    print(ind, anno)




