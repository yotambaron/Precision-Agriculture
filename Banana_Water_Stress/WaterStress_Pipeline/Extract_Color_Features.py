import os
import cv2
import numpy as np
import itertools
import pandas as pd


def VARI(blues, greens, reds):
    return (greens - reds) / (greens + reds - blues)


def NDI(blues, greens, reds):
    green_norm = greens / (blues + greens + reds)
    red_norm = reds / (blues + greens + reds)
    return (red_norm - green_norm) / (red_norm + green_norm)


def EGI(blues, greens, reds):
    blue_norm = blues / (blues + greens + reds)
    green_norm = greens / (blues + greens + reds)
    red_norm = reds / (blues + greens + reds)
    return 2 * green_norm - red_norm - blue_norm


def get_aggregated_leaf_indices(indices_array):
    try:
        avg_leaves = [np.mean(leaf) for leaf in indices_array]  # Average hues for each plant's leaf
        std_leaves = [np.std(leaf) for leaf in indices_array]  # Std hues for each plant's leaf
        range_leaves = [np.max(leaf) - np.min(leaf) for leaf in indices_array]  # Range hues for each plant's leaf
        return avg_leaves, std_leaves, range_leaves
    except:
        return np.repeat(-1, 3)


def remove_color_outliers(colors):
    quantile_10 = np.quantile(list(colors), 0.5 / 10)  # Get the 10th angles quantile
    quantile_90 = np.quantile(list(colors), 9.5 / 10)  # Get the 90th angles quantile
    colors = [x for x in colors if x >= quantile_10]
    colors = [x for x in colors if x <= quantile_90]
    colors = list(colors)
    return colors


def compute_hue_vari_ndi_egi(rgb_matrix, leaves_anno_list, path):

    hsv_image = cv2.cvtColor(rgb_matrix.copy(), cv2.cv2.COLOR_BGR2HSV)
    hues_plant = []
    varis_plant = []
    ndis_plant = []
    egis_plant = []
    saturations = []  # HSV - S
    values = []  # HSV - V

    # Go through the plant's leaves
    for ind, txt in enumerate(leaves_anno_list):
        blues = []
        greens = []
        reds = []
        hsvs = []   # HSV - Hue (H)

        if txt.endswith('points.txt'):  # Only files that are the points inside the leaf's polygon
            curr_path = path + '/' + txt
            points = np.loadtxt(curr_path, delimiter=',')  # Specific leaf annotation
            # Go through all the points inside the polygon and calculate their RGB and hue values
            try:
                for i in range(points.shape[0]):
                    height = int(points[i, 0])
                    width = int(points[i, 1])
                    blues.append(int(rgb_matrix[height, width, 0]))
                    greens.append(int(rgb_matrix[height, width, 1]))
                    reds.append(int(rgb_matrix[height, width, 2]))
                    hsvs.append(hsv_image[height, width, 0] * 2)
                    saturations.append(hsv_image[height, width, 1] / 255)
                    values.append(hsv_image[height, width, 2] / 255)

                # Turn colors from list to an array (for technical computation reasons)
                blues = np.array(blues)
                greens = np.array(greens)
                reds = np.array(reds)
                # Remove outliers
                hsvs = remove_color_outliers(hsvs)
                saturations = remove_color_outliers(saturations)
                values = remove_color_outliers(values)
                # Add the leaf's indices to the plant's indices array
                hues_plant.append(hsvs)
                varis_plant.append(VARI(blues, greens, reds))
                ndis_plant.append(NDI(blues, greens, reds))
                egis_plant.append(EGI(blues, greens, reds))
            except:
                print("Not enough points in the annotation: ", txt)

    return hues_plant, saturations, values, varis_plant, ndis_plant, egis_plant


def color_compute_all():

    # Set path and all treatments and dates
    leaves_anno_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
    image_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\RGB'
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

    # Whole plant features
    df['Avg_hue'], df['Avg_std_hue'], df['Avg_range_hue'], \
    df['Std_avg_hue'], df['Std_std_hue'], df['Std_range_hue'], \
    df['Avg_saturation'], df['Avg_value_hsv'], \
    df['Avg_vari'], df['Std_vari'], df['Range_vari'], \
    df['Avg_ndi'], df['Std_ndi'], df['Range_ndi'], \
    df['Avg_egi'], df['Std_egi'], df['Range_egi'] = np.repeat(-1, 17)

    # Min leaf features
    df['Avg_hue_min_leaf'], df['Std_hue_min_leaf'], df['Range_hue_min_leaf'], \
    df['Avg_vari_min_leaf'], df['Std_vari_min_leaf'], df['Range_vari_min_leaf'], \
    df['Avg_ndi_min leaf'], df['Std_ndi_min_leaf'], df['Range_ndi_min_leaf'], \
    df['Avg_egi_min_leaf'], df['Std_egi_min_leaf'], df['Range_egi_min_leaf'] = np.repeat(-1, 12)

    # Second min leaf features
    df['Avg_hue_2min_leaf'], df['Std_hue_2min_leaf'], df['Range_hue_2min_leaf'], \
    df['Avg_vari_2min_leaf'], df['Std_vari_2min_leaf'], df['Range_vari_2min_leaf'], \
    df['Avg_ndi_2min leaf'], df['Std_ndi_2min_leaf'], df['Range_ndi_2min_leaf'], \
    df['Avg_egi_2min_leaf'], df['Std_egi_2min_leaf'], df['Range_egi_2min_leaf'] = np.repeat(-1, 12)

    dir_list = os.listdir(leaves_anno_path)
    # Go through all the plants
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
        rgb_matrix = cv2.imread(image_path + '/' + folder + '.JPEG')
        leaves_anno_list = os.listdir(leaves_anno_path + '/' + folder)
        plant_path = leaves_anno_path + '/' + folder

        # Get indices of the plant by its leaves
        [hues_plant, saturations, values, varis_plant, ndis_plant, egis_plant] = \
            compute_hue_vari_ndi_egi(rgb_matrix, leaves_anno_list, plant_path)

        # Get averages, stds and ranges of all plant's leaves over all the indices
        [avg_hues_plant, std_hues_plant, range_hues_plant] = get_aggregated_leaf_indices(hues_plant)
        [avg_varis_plant, std_varis_plant, range_varis_plant] = get_aggregated_leaf_indices(varis_plant)
        [avg_ndis_plant, std_ndis_plant, range_ndis_plant] = get_aggregated_leaf_indices(ndis_plant)
        [avg_egis_plant, std_egis_plant, range_egis_plant] = get_aggregated_leaf_indices(egis_plant)

        try:
            # Assign whole plant features to the data frame
            df.at[index, 'Avg_hue'] = np.mean(avg_hues_plant)
            df.at[index, 'Avg_std_hue'] = np.mean(std_hues_plant)
            df.at[index, 'Avg_range_hue'] = np.mean(range_hues_plant)
            df.at[index, 'Std_avg_hue'] = np.std(avg_hues_plant)
            df.at[index, 'Std_std_hue'] = np.std(std_hues_plant)
            df.at[index, 'Std_range_hue'] = np.std(range_hues_plant)
            df.at[index, 'Avg_saturation'] = np.mean(saturations)
            df.at[index, 'Avg_value_hsv'] = np.mean(values)
            df.at[index, 'Avg_vari'] = np.mean(avg_varis_plant)
            df.at[index, 'Std_vari'] = np.mean(std_varis_plant)
            df.at[index, 'Range_vari'] = np.mean(range_varis_plant)
            df.at[index, 'Avg_ndi'] = np.mean(avg_ndis_plant)
            df.at[index, 'Std_ndi'] = np.mean(std_ndis_plant)
            df.at[index, 'Range_ndi'] = np.mean(range_ndis_plant)
            df.at[index, 'Avg_egi'] = np.mean(avg_egis_plant)
            df.at[index, 'Std_egi'] = np.mean(std_egis_plant)
            df.at[index, 'Range_egi'] = np.mean(range_egis_plant)

            # Get indices of the min leaf
            hues_min_leaf = hues_plant[min_leaf]
            varis_min_leaf = varis_plant[min_leaf]
            ndis_min_leaf = ndis_plant[min_leaf]
            egis_min_leaf = egis_plant[min_leaf]

            # Get indices of the second min leaf
            hues_2min_leaf = hues_plant[second_min_leaf]
            varis_2min_leaf = varis_plant[second_min_leaf]
            ndis_2min_leaf = ndis_plant[second_min_leaf]
            egis_2min_leaf = egis_plant[second_min_leaf]

            # Assign min leaf features to the data frame
            df.at[index, 'Avg_hue_min_leaf'] = np.mean(hues_min_leaf)
            df.at[index, 'Std_hue_min_leaf'] = np.std(hues_min_leaf)
            df.at[index, 'Range_hue_min_leaf'] = np.max(hues_min_leaf) - np.min(hues_min_leaf)
            df.at[index, 'Avg_vari_min_leaf'] = np.mean(varis_min_leaf)
            df.at[index, 'Std_vari_min_leaf'] = np.std(varis_min_leaf)
            df.at[index, 'Range_vari_min_leaf'] = np.max(varis_min_leaf) - np.min(varis_min_leaf)
            df.at[index, 'Avg_ndi_min leaf'] = np.mean(ndis_min_leaf)
            df.at[index, 'Std_ndi_min_leaf'] = np.std(ndis_min_leaf)
            df.at[index, 'Range_ndi_min_leaf'] = np.max(ndis_min_leaf) - np.min(ndis_min_leaf)
            df.at[index, 'Avg_egi_min_leaf'] = np.mean(egis_min_leaf)
            df.at[index, 'Std_egi_min_leaf'] = np.std(egis_min_leaf)
            df.at[index, 'Range_egi_min_leaf'] = np.max(egis_min_leaf) - np.min(egis_min_leaf)

            # Assign second min leaf features to the data frame
            df.at[index, 'Avg_hue_2min_leaf'] = np.mean(hues_2min_leaf)
            df.at[index, 'Std_hue_2min_leaf'] = np.std(hues_2min_leaf)
            df.at[index, 'Range_hue_2min_leaf'] = np.max(hues_2min_leaf) - np.min(hues_2min_leaf)
            df.at[index, 'Avg_vari_2min_leaf'] = np.mean(varis_2min_leaf)
            df.at[index, 'Std_vari_2min_leaf'] = np.std(varis_2min_leaf)
            df.at[index, 'Range_vari_2min_leaf'] = np.max(varis_2min_leaf) - np.min(varis_2min_leaf)
            df.at[index, 'Avg_ndi_2min leaf'] = np.mean(ndis_2min_leaf)
            df.at[index, 'Std_ndi_2min_leaf'] = np.std(ndis_2min_leaf)
            df.at[index, 'Range_ndi_2min_leaf'] = np.max(ndis_2min_leaf) - np.min(ndis_2min_leaf)
            df.at[index, 'Avg_egi_2min_leaf'] = np.mean(egis_2min_leaf)
            df.at[index, 'Std_egi_2min_leaf'] = np.std(egis_2min_leaf)
            df.at[index, 'Range_egi_2min_leaf'] = np.max(egis_2min_leaf) - np.min(egis_2min_leaf)
        except:
            print("Min leaf problem due to too small annotations: ", folder)

        # Save the dataframe
        df.to_csv(save_path + '/Features-color_hue_new.csv', index=False)
        print("Finished computing plant: ", folder)


if __name__ == "__main__":
    color_compute_all()










