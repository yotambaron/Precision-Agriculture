from shapely.geometry.polygon import Polygon
import os
import numpy as np
from PIL import Image, ImageDraw
import shapely.speedups
shapely.speedups.enable()


def switch_x_y(annotation):
    text_list = list(annotation)
    for i, el in enumerate(text_list):
        text_list[i] = (el[1], el[0])
    return text_list


# Draw leaves and plant annotations on image and save only the leaves that are inside the plant
image_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Segmentation_try\leafsegmentor\inference_output\512_plant_segmentor_inference\clean'
plant_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Plant\New_Segmentor_Annotations'
leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves\New_Segmentor_Annotations'
dest_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'

for im in os.listdir(image_path):
    if im.lower().endswith('.jpeg'):
        image = Image.open(image_path + '/' + im)
        image2 = image.copy()
        folder_name = im[:-5]

        curr_plant_anno = plant_anno_path + '/' + folder_name
        curr_leaves_anno = leaves_anno_path + '/' + folder_name

        plant_poly_array = []
        leaves_anno_array = []

        try:
            # Save all current plant's annotations in an array as polygons
            for plant in os.listdir(curr_plant_anno):
                text = np.loadtxt(curr_plant_anno + '/' + plant, delimiter=',')
                text = switch_x_y(text)
                temp_poly = Polygon(text)
                plant_poly_array.append(temp_poly)
        except:
            print("No plant annotations: ", folder_name)

        try:
            # Save all current leaves' annotations in an array
            for leaf in os.listdir(curr_leaves_anno):
                text = np.loadtxt(curr_leaves_anno + '/' + leaf, delimiter=',')
                text = switch_x_y(text)
                leaves_anno_array.append(text)
        except:
            print("No leaves annotations: ", folder_name)

        try:
            # Check if the leaf's centroid is inside the plant's polygon, draw it on the image and save the leaf's annotations
            for ind, leaf_anno in enumerate(leaves_anno_array):
                leaf_poly = Polygon(leaf_anno)
                for ind2, plant_poly in enumerate(plant_poly_array):
                    if leaf_poly.centroid.within(plant_poly):
                        draw = ImageDraw.Draw(image2)
                        draw.polygon(leaf_anno, outline="black")
                        leaf_anno = np.array(leaf_anno)
                        np.savetxt(dest_path + '/' + folder_name + '/' + folder_name + '_' + str(ind) + '.txt', leaf_anno, delimiter=',')
        except:
            print("No plant/leaves annotations: ", folder_name)

        # Save image with the plant and leaves annotations
        image2.save(dest_path + '/' + folder_name + '/' + im, 'JPEG')
        print("Finished plant: ", folder_name)









