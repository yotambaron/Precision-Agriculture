import os
import numpy as np
import itertools
import pandas as pd
from sympy import Polygon


def create_polygon(annotations):
    points = []
    if len(annotations) > 2:
        for i in range(annotations.shape[0]):
            point = (annotations[i, 0], annotations[i, 1])
            points.append(point)
        polygon = Polygon(*points)
        return polygon
    else:
        return np.nan


def clean_angles(angles):
    quantile_10 = np.quantile(list(angles), 1 / 10)  # Get the 10th angles quantile
    quantile_90 = np.quantile(list(angles), 9 / 10)  # Get the 90th angles quantile
    angles = [x for x in angles if x >= quantile_10]
    angles = [x for x in angles if x <= quantile_90]
    angles = np.array(list(map(float, angles))) * 180/np.pi
    return angles


def compute_plant_geometric(leaves_anno_list, path):

    sizes = []
    perimeters = []
    avg_angles = []
    all_angles = []
    polar_sec_moment_of_areas = []

    # Create plant's polygons from plant's leaves and calculate area, perimeter and angles
    for ind, txt in enumerate(leaves_anno_list):
        if not txt.endswith('JPEG'):  # Leaves only
            curr_path = path + '/' + txt
            annotations = np.loadtxt(curr_path, delimiter=',')  # Specific leaf annotation
            leaf_poly = create_polygon(annotations)
            if not type(leaf_poly) == float:    # Check this reason
                try:
                    sizes.append(float(leaf_poly.area))
                    perimeters.append(float(leaf_poly.perimeter))
                    angles = clean_angles(leaf_poly.angles.values())  # Remove 10th and 90th quantile and make degrees
                    avg_angles.append(np.mean(angles))
                    all_angles.append(angles)
                    polar_sec_moment_of_areas.append(float(leaf_poly.polar_second_moment_of_area()))
                except:
                    print("Computation Error ", txt)

    return sizes, perimeters, avg_angles, all_angles, polar_sec_moment_of_areas


def geometric_compute_all():

    # Set path and all treatments and dates
    leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
    save_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'

    treatments = ['A', 'B', 'C', 'D']
    num_plants = range(1, 49)
    dates = range(1, 42)
    dates_dic = {'task_767': '1', 'task_768': '2', 'task_769': '3', 'task_770': '4', 'task_771': '5', 'task_772': '6',
                 'task_774': '7', 'task_775': '8', 'task_776': '9', 'task_777': '10', 'task_778': '11',
                 'task_779': '12', 'task_780': '13', 'task_781': '14', 'task_782': '15', 'task_783': '16',
                 'task_784': '17', 'task_790': '18', 'task_791': '19', 'task_792': '20', 'task_793': '21',
                 'task_801': '22', 'task_802': '23', 'task_803': '24', 'task_804': '25', 'task_805': '26',
                 'task_807': '27', 'task_808': '28', 'task_809': '29', 'task_810': '30', 'task_812': '31',
                 'task_813': '32', 'task_814': '33', 'task_815': '34', 'task_817': '35', 'task_818': '36',
                 'task_819': '37', 'task_820': '38', 'task_821': '39', 'task_822': '40', 'task_823': '41'}

    # Create all plants and dates combination
    a = [treatments, num_plants, dates]
    data = list(itertools.product(*a))
    idx = ['c{}'.format(i) for i in range(1, len(data) + 1)]

    # Data frame with all wanted features
    df = pd.DataFrame(data, index=idx, columns=['Treatment', 'Num', 'Date'])
    df['Avg_size'] = -1
    df['Max_size'] = -1
    df['Min_size'] = -1
    df['Std_size'] = -1

    df['Avg_perimeter'] = -1
    df['Max_perimeter'] = -1
    df['Min_perimeter'] = -1
    df['Std_perimeter'] = -1

    df['Avg_plant_angles'] = -1
    df['Std_plant_angles'] = -1
    df['Max_avg_leaves_angles'] = -1
    df['Std_avg_leaves_angles'] = -1

    df['Avg_polar_second_moment_area'] = -1

    # Go through each plant's leaves annotation to compute indices
    dir_list = os.listdir(leaves_anno_path)
    for ind, folder in enumerate(dir_list):

        # Find the current index in the data frame
        treatment = folder[9]
        date = int(dates_dic[folder[:8]])
        plant_num = int(folder[12]) if folder[11] == 0 else int(folder[11:13])
        index = df.index[(df['Treatment'] == treatment) & (df['Num'] == plant_num) & (df['Date'] == date)]

        # Get plant's indices
        leaves_anno_list = os.listdir(leaves_anno_path + '/' + folder)
        plant_path = leaves_anno_path + '/' + folder
        [sizes, perimeters, avg_angles, all_angles, polar_sec_moment] = compute_plant_geometric(leaves_anno_list, plant_path)

        # Save area indices to the data frame
        df.at[index, 'Avg_size'] = np.mean(sizes)
        df.at[index, 'Max_size'] = np.max(sizes)
        df.at[index, 'Min_size'] = np.min(sizes)
        df.at[index, 'Std_size'] = np.std(sizes)

        # Save perimeter indices to the data frame
        df.at[index, 'Avg_perimeter'] = np.mean(perimeters)
        df.at[index, 'Max_perimeter'] = np.max(perimeters)
        df.at[index, 'Min_perimeter'] = np.min(perimeters)
        df.at[index, 'Std_perimeter'] = np.std(perimeters)

        # Save angles indices to the data frame
        all_angles = [item for sublist in all_angles for item in sublist]  # Flatten all temps from all leaves
        df.at[index, 'Avg_plant_angles'] = np.mean(all_angles)
        df.at[index, 'Std_plant_angles'] = np.std(all_angles)
        df.at[index, 'Max_avg_leaves_angles'] = np.max(avg_angles)
        df.at[index, 'Std_avg_leaves_angles'] = np.std(avg_angles)

        # Save area moments indices to the data frame
        df.at[index, 'Avg_polar_second_moment_area'] = np.mean(polar_sec_moment)

        # Save the dataframe
        df.to_csv(save_path + '/Features-geometric.csv', index=False)
        print("Finished computing plant: ", folder)


if __name__ == "__main__":
    geometric_compute_all()

