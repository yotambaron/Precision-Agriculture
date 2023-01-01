import os
import cv2
import itertools
import pandas as pd
import numpy as np


def compute_plant_thermals(thermal_matrix, leaves_anno_list, path, min_leaf, second_min_leaf):

    thermals_array = []
    avg_thermals_array = []
    stds_array = []

    # Get plant's leaves temps
    for ind, txt in enumerate(leaves_anno_list):
        leaf_thermals = []
        if txt.endswith('points.txt'):  # Only files that are the points inside the leaf's polygon
            curr_path = path + '/' + txt
            points = np.loadtxt(curr_path, delimiter=',')  # Specific leaf points
            # Save all the temps of points inside the leaf
            for i in range(points.shape[0]):
                width = int(points[i, 0])
                height = int(points[i, 1])
                leaf_thermals.append(thermal_matrix[width, height])

            thermals_array.append(leaf_thermals)  # Save all thermals of the specific leaf in an array
            avg_thermals_array.append(np.mean(leaf_thermals))  # Save the avg thermal of the specific leaf in an array
            stds_array.append(np.std(leaf_thermals))  # Save the std of the specific leaf in an array

    try:
        # Compute indices for whole plant
        flat_temps_array = [item for sublist in thermals_array for item in sublist]  # Flatten all temps from all leaves
        third_quantile = np.quantile(flat_temps_array, 1 / 3)
        third_temps = [x for x in flat_temps_array if x <= third_quantile]

        avg_plant = np.mean(flat_temps_array)   # Average of all the pixels of the plant
        std_plant = np.std(flat_temps_array)    # Std of all the pixels of the plant
        avg_leaves = np.mean(avg_thermals_array)    # Average of the average temps of all leaves of the plant
        max_leaves = np.max(avg_thermals_array)    # Leaf with maximum average temp
        min_leaves = np.min(avg_thermals_array)    # Leaf with minimum average temp
        std_leaves = np.std(avg_thermals_array)    # Std of the average temps of all leaves of the plant
        avg_plant_third = np.mean(third_temps)     # The average temp of the third lowest temps of the plant
        std_plant_third = np.std(third_temps)   # The std of the third lowest temps of the plant

        # Compute indices for min leaf
        min_leaf_thermals = thermals_array[min_leaf]
        min_third_quantile = np.quantile(min_leaf_thermals, 1 / 3)  # Get the third lowest quantile temps
        min_third_temps = [x for x in min_leaf_thermals if x <= min_third_quantile]
        avg_min_leaf = avg_thermals_array[min_leaf]
        max_min_leaf = np.max(min_leaf_thermals)
        min_min_leaf = np.min(min_leaf_thermals)
        std_min_leaf = np.std(min_leaf_thermals)
        avg_third_min_leaf = np.mean(min_third_temps)
        std_third_min_leaf = np.std(min_third_temps)

        # Compute indices for second min leaf
        second_min_leaf_thermals = thermals_array[second_min_leaf]
        min2_third_quantile = np.quantile(second_min_leaf_thermals, 1 / 3)
        min2_third_temps = [x for x in second_min_leaf_thermals if x <= min2_third_quantile]
        avg_2min_leaf = avg_thermals_array[second_min_leaf]
        max_2min_leaf = np.max(second_min_leaf_thermals)
        min_2min_leaf = np.min(second_min_leaf_thermals)
        std_2min_leaf = np.std(second_min_leaf_thermals)
        avg_third_2min_leaf = np.mean(min2_third_temps)
        std_third_2min_leaf = np.std(min2_third_temps)

    except:
        print("Empty thermal image: ", txt)
        return np.repeat(-1, 20)

    if np.size(avg_thermals_array) == 1:
        print("Only one leaf: ", path)
        max_plant, min_plant, std_plant, \
        avg_2min_leaf, max_2min_leaf, min_2min_leaf, std_2min_leaf, avg_third_2min_leaf, std_third_2min_leaf \
            = np.repeat(-1, 9)

    return avg_plant, std_plant, avg_leaves, max_leaves, min_leaves, std_leaves, avg_plant_third, std_plant_third, \
           avg_min_leaf, max_min_leaf, min_min_leaf, std_min_leaf, avg_third_min_leaf, std_third_min_leaf, \
           avg_2min_leaf, max_2min_leaf, min_2min_leaf, std_2min_leaf, avg_third_2min_leaf, std_third_2min_leaf


def thermal_compute_all():
    # Set path and all treatments and dates
    leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
    thermal_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\Thermal'
    save_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
    min_leaves_df = pd.read_csv(r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\depth_min_2ndmin_leaves.csv')

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

    # Whole plant indices
    df['Avg_plant_temp'], df['Std_plant_temp'], df['Avg_leaves_temp'], df['Max_leaves_temp'], df['Min_leaves_temp'], \
    df['Std_leaves_temp'], df['Avg_plant_third_temp'], df['Std_plant_third_temp'] = np.repeat(-1, 8)

    # Min leaf indices
    df['Avg_min_leaf'], df['Max_min_leaf'], df['Min_min_leaf'], df['Std_min_leaf'], \
    df['Avg_third_min_leaf'], df['Std_third_min_leaf'] = np.repeat(-1, 6)

    # Second min leaf indices
    df['Avg_2ndmin_leaf'], df['Max_2ndmin_leaf'], df['Min_2ndmin_leaf'], df['Std_2ndmin_leaf'], \
    df['Avg_third_2ndmin_leaf'], df['Std_third_2ndmin_leaf'] = np.repeat(-1, 6)

    dir_list = os.listdir(leaves_anno_path)
    for ind, folder in enumerate(dir_list):
        # Find the current index in the data frame
        treatment = folder[9]
        date = int(dates_dic[folder[:8]])
        plant_num = int(folder[12]) if folder[11] == 0 else int(folder[11:13])
        index = df.index[(df['Treatment'] == treatment) & (df['Num'] == plant_num) & (df['Date'] == date)]

        # Extract the position of min and second min leaves
        min_leaves_index = index[0]
        min_leaves_index = int(min_leaves_index[1:len(min_leaves_index)]) - 1
        min_leaf = int(min_leaves_df.at[min_leaves_index, 'Min_Leaf'])  # Maybe str and not int?
        second_min_leaf = int(min_leaves_df.at[min_leaves_index, '2nd_Min_Leaf'])

        # Compute thermal indices for each plant (and find min and second min leaves)
        thermal_matrix = cv2.imread(thermal_path + '/' + folder + '.PNG', -1)
        leaves_anno_list = os.listdir(leaves_anno_path + '/' + folder)
        plant_path = leaves_anno_path + '/' + folder

        [avg_plant, std_plant, avg_leaves, max_leaves, min_leaves, std_leaves, avg_plant_third, std_plant_third,
         avg_min_leaf, max_min_leaf, min_min_leaf, std_min_leaf, avg_third_min_leaf, std_third_min_leaf,
         avg_2min_leaf, max_2min_leaf, min_2min_leaf, std_2min_leaf, avg_third_2min_leaf, std_third_2min_leaf] = \
            compute_plant_thermals(thermal_matrix, leaves_anno_list, plant_path, min_leaf, second_min_leaf)

        # Save to a data frame whole plant's indices
        df.at[index, 'Avg_plant_temp'] = avg_plant
        df.at[index, 'Std_plant_temp'] = std_plant
        df.at[index, 'Avg_leaves_temp'] = avg_leaves
        df.at[index, 'Max_leaves_temp'] = max_leaves
        df.at[index, 'Min_leaves_temp'] = min_leaves
        df.at[index, 'Std_leaves_temp'] = std_leaves
        df.at[index, 'Avg_plant_third_temp'] = avg_plant_third
        df.at[index, 'Std_plant_third_temp'] = std_plant_third

        # Save to a data frame min leaf indices
        df.at[index, 'Avg_min_leaf'] = avg_min_leaf
        df.at[index, 'Max_min_leaf'] = max_min_leaf
        df.at[index, 'Min_min_leaf'] = min_min_leaf
        df.at[index, 'Std_min_leaf'] = std_min_leaf
        df.at[index, 'Avg_third_min_leaf'] = avg_third_min_leaf
        df.at[index, 'Std_third_min_leaf'] = std_third_min_leaf

        # Save to a data frame second min leaf indices
        df.at[index, 'Avg_2ndmin_leaf'] = avg_2min_leaf
        df.at[index, 'Max_2ndmin_leaf'] = max_2min_leaf
        df.at[index, 'Min_2ndmin_leaf'] = min_2min_leaf
        df.at[index, 'Std_2ndmin_leaf'] = std_2min_leaf
        df.at[index, 'Avg_third_2ndmin_leaf'] = avg_third_2min_leaf
        df.at[index, 'Std_third_2ndmin_leaf'] = std_third_2min_leaf

        # Save the dataframe
        df.to_csv(save_path + '/Features-thermal.csv', index=False)
        print("Finished computing plant: ", folder)


if __name__ == "__main__":
    thermal_compute_all()
