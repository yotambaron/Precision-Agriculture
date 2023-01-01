import numpy as np
import os
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


def switch_x_y(annotation):
    text_list = list(annotation)
    for i, el in enumerate(text_list):
        text_list[i] = (el[1], el[0])
    return text_list


def get_points_in_poly(annotation, poly):
    points_in_poly = []
    x_min = min(annotation[:, 1])
    x_max = max(annotation[:, 1])
    y_min = min(annotation[:, 0])
    y_max = max(annotation[:, 0])
    for width in range(int(x_min), int(x_max) + 1):
        for height in range(int(y_min), int(y_max) + 1):
            point = Point(width, height)
            if poly.contains(point):  # if pixel is inside the leaf annotations
                points_in_poly.append([width, height])
    return points_in_poly


leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'

dir_list = os.listdir(leaves_anno_path)
for ind, folder in enumerate(dir_list):

    leaves_anno_list = os.listdir(leaves_anno_path + '/' + folder)
    plant_path = leaves_anno_path + '/' + folder

    for i, txt in enumerate(leaves_anno_list):
        if not txt.endswith('JPEG'):  # Leaves only
            curr_path = plant_path + '/' + txt
            annotations = np.loadtxt(curr_path, delimiter=',')  # Specific leaf annotation
            temp_annotations = switch_x_y(annotations)
            temp_leaf_poly = Polygon(temp_annotations)  # Make a polygon of the annotations
            points = get_points_in_poly(annotations, temp_leaf_poly)    # Get all points inside the polygon
            np.savetxt(leaves_anno_path + '/' + folder + '/' + txt[:-4] + '_points.txt', points, delimiter=',')

    print("Finished computing points in plant: ", folder)
