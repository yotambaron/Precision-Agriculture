import os
import cv2
import itertools
import pandas as pd
import numpy as np


def compute_depth_indices(depth_matrix, leaves_anno_list, path):

    depths_array = []
    stds_array = []

    # Get plant's leaves depths
    for ind, txt in enumerate(leaves_anno_list):

        leaf_depths = []
        if txt.endswith('points.txt'):  # Only files that are the points inside the leaf's polygon
            curr_path = path + '/' + txt
            points = np.loadtxt(curr_path, delimiter=',')  # Specific leaf points
            # Save all the depths of points inside the leaf
            for i in range(points.shape[0]):
                width = int(points[i, 0])
                height = int(points[i, 1])
                if depth_matrix[width, height] > 500:
                    leaf_depths.append(depth_matrix[width, height])

            depths_array.append(np.mean(leaf_depths))   # Save the depth of the specific leaf in an array
            stds_array.append(np.std(leaf_depths))  # Save the std of the specific leaf in an array

    # Get the position of the min and second min leaves
    min_leaf = np.argmin(depths_array)
    temp_depths_array = depths_array.copy()
    temp_depths_array[min_leaf] = 1e7
    second_min_leaf = np.argmin(temp_depths_array)

    # Compute average indices
    avg_all = np.mean(depths_array)
    avg_min = depths_array[min_leaf]
    avg_2min = depths_array[second_min_leaf]

    # Compute std indices
    std_avg_all = np.std(depths_array)
    avg_std_all = np.mean(stds_array)
    std_min = stds_array[min_leaf]
    std_2min = stds_array[second_min_leaf]

    if np.size(depths_array) == 1:
        print("Only one leaf: ", path)
        return min_leaf, -1, avg_all, avg_min, -1, -1, avg_std_all, std_min, -1

    return min_leaf, second_min_leaf, avg_all, avg_min, avg_2min, std_avg_all, avg_std_all, std_min, std_2min


def depth_compute_all():
    # Set path and all treatments and dates
    leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
    depth_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Depth'
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
    df['Avg_Depth_All_Leaves'] = -1
    df['Avg_Depth_Min_Leaf'] = -1
    df['Avg_Depth_2nd_Min_Leaf'] = -1
    df['Std_Avg_Depth_All_Leaves'] = -1
    df['Avg_Std_Depth_All_Leaves'] = -1
    df['Std_Depth_Min_Leaf'] = -1
    df['Std_Depth_2nd_Min_Leaf'] = -1

    # Data frame with all plants' min and 2nd min depth leaves
    df_min = pd.DataFrame(data, index=idx, columns=['Treatment', 'Num', 'Date'])
    df_min['Min_Leaf'] = -1
    df_min['2nd_Min_Leaf'] = -1

    dir_list = os.listdir(leaves_anno_path)
    for ind, folder in enumerate(dir_list):
        # Find the current index in the data frame
        treatment = folder[9]
        date = int(dates_dic[folder[:8]])
        plant_num = int(folder[12]) if folder[11] == 0 else int(folder[11:13])
        index = df.index[(df['Treatment'] == treatment) & (df['Num'] == plant_num) & (df['Date'] == date)]

        # Compute depth indices for each plant (and find min and second min leaves)
        depth_matrix = cv2.imread(depth_path + '/' + folder + '.PNG', -1)
        leaves_anno_list = os.listdir(leaves_anno_path + '/' + folder)
        plant_path = leaves_anno_path + '/' + folder
        [min_leaf, second_min_leaf, avg_all, avg_min, avg_2min, std_avg_all, avg_std_all, std_min, std_2min] = \
            compute_depth_indices(depth_matrix, leaves_anno_list, plant_path)

        # Save to data frames
        df.at[index, 'Avg_Depth_All_Leaves'] = avg_all
        df.at[index, 'Avg_Depth_Min_Leaf'] = avg_min
        df.at[index, 'Avg_Depth_2nd_Min_Leaf'] = avg_2min
        df.at[index, 'Std_Avg_Depth_All_Leaves'] = std_avg_all
        df.at[index, 'Avg_Std_Depth_All_Leaves'] = avg_std_all
        df.at[index, 'Std_Depth_Min_Leaf'] = std_min
        df.at[index, 'Std_Depth_2nd_Min_Leaf'] = std_2min

        df_min.at[index, 'Min_Leaf'] = min_leaf
        df_min.at[index, '2nd_Min_Leaf'] = second_min_leaf

        # Save the dataframe
        df.to_csv(save_path + '/Features-depth.csv', index=False)
        df_min.to_csv(save_path + '/depth_min_2ndmin_leaves.csv', index=False)
        print("Finished computing plant: ", folder)


if __name__ == "__main__":
    depth_compute_all()
