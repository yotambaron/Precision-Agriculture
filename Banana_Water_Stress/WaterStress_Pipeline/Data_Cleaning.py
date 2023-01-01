import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------------------------------------- #
# ------ Functions for switching between B&C treatments, removing plants, completing missing data and smoothing ------ #
# -------------------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------ Switch treatments  ------------------------------------------------ #
def switch_b_c(df):
    df.at[df['Treatment'] == 'B', 'Treatment'] = 'R'
    df.at[df['Treatment'] == 'C', 'Treatment'] = 'B'
    df.at[df['Treatment'] == 'R', 'Treatment'] = 'C'
    return df


# ------------------------------------------------- Removing plants  ------------------------------------------------- #
def plants_to_remove(plant_nums, plant_treatments, df, column, missing_days):
    plants_remove = []
    for n in plant_nums:
        for treat in plant_treatments:
            df_cur = df[(df['Treatment'] == treat) & (df['Num'] == n)]
            missing_inds = np.where(df_cur[column] == -1)[0]
            flag = False
            for m in range(0, np.size(missing_inds) - 2):
                if (missing_inds[m + missing_days] - missing_inds[m] == missing_days) | (np.size(missing_inds) > 5):
                    flag = True
            if flag:
                plants_remove.append(treat + str(n))
    return plants_remove


# --------------------------------------------- Completing missing values  ------------------------------------------- #
def complete_missing_values(DF, with_index=False):
    if with_index:
        start_column = 3
    else:
        start_column = 2

    # Complete missing values for the last date (41) with the one before (40) or two before (39)
    df_temp = DF.reset_index(drop=True)
    for i in range(df_temp.shape[0]):
        if df_temp['Date'][i] == 41:
            for ind, col in enumerate(df_temp.columns):
                if (ind > start_column) and (df_temp.at[i, col] == -1):
                    if df_temp[col][i - 1] != -1:   # if day before last is not also missing
                        df_temp.at[i, col] = df_temp[col][i - 1]
                    else:   # if day before last is missing, fill with two days before last
                        df_temp.at[i, col] = df_temp[col][i - 2]

    # Complete missing values for the first date (1) with the second (2) or with the third (3)
    for i in range(df_temp.shape[0]):
        if df_temp['Date'][i] == 1:
            for ind, col in enumerate(df_temp.columns):
                if (ind > start_column) and (df_temp.at[i, col] == -1):
                    if df_temp[col][i + 1] != -1:  # if second day is not also missing
                        df_temp.at[i, col] = df_temp[col][i + 1]
                    else:  # if second is missing, fill with the third
                        df_temp.at[i, col] = df_temp[col][i + 2]

    # Complete the other cases - complete with the mean of the day before and after that have values
    for i in range(df_temp.shape[0]):
        if (df_temp['Date'][i] != 1) and (df_temp['Date'][i] != 41):
            for ind, col in enumerate(df_temp.columns):
                if (ind > start_column) and (df_temp.at[i, col] == -1):
                    if df_temp[col][i + 1] != -1:   # if following day is not also missing
                        df_temp.at[i, col] = (df_temp[col][i - 1] + df_temp[col][i + 1]) / 2
                    else:   # if following day is missing, fill with two days after
                        df_temp.at[i, col] = (df_temp[col][i - 1] + df_temp[col][i + 2]) / 2
    return df_temp


# ------------------------------------------------ Linear Smoothing  ------------------------------------------------- #
def roll(series, window):
    rolled = series.rolling(window=window).mean()
    for w in range(window-1):
        rolled[rolled.index[w]] = series[series.index[w]]
    return rolled


def smooth_df(nums_plants, treatments_plants, df, col_start, deleted):
    smoothed_df = pd.DataFrame()
    for ind, t in enumerate(treatments_plants):
        for n in nums_plants:
            delete = t + str(n)
            if not (delete in deleted):
                df_temp = df.copy()[(df['Treatment'] == t) & (df['Num'] == n)]
                for i, col in enumerate(df.columns):
                    if i > col_start:
                        df_temp.at[:, col] = roll(df_temp[col], window=3)
                smoothed_df = pd.concat([smoothed_df, df_temp])
    return smoothed_df


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------- Option 1 ------------------------------------------------------- #
# ---------------------- Clean all features separately and then merge all to one data frame -------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

#  Load all indices files
features_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
save_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
df_color = pd.read_csv(features_path + '/Features-color.csv')
df_geometric = pd.read_csv(features_path + '/Features-geometric.csv')
df_depth = pd.read_csv(features_path + '/Features-depth.csv')
df_thermal = pd.read_csv(features_path + '/Features-thermal.csv')
df_leavesNum = pd.read_csv(features_path + '/Features-num_of_leaves.csv')

# Switch treatments for all indices files
df_color = switch_b_c(df_color.copy())
df_geometric = switch_b_c(df_geometric.copy())
df_depth = switch_b_c(df_depth.copy())
df_thermal = switch_b_c(df_thermal.copy())
df_leeavesNum = switch_b_c(df_leavesNum.copy())

# Find plants to remove for each indices file
nums = range(1, 49)
treatments = ['A', 'B', 'C', 'D']
days_missing = 2
remove_color = plants_to_remove(nums, treatments, df_color, 'Mean green min leaf', days_missing)
remove_geometric = plants_to_remove(nums, treatments, df_geometric, 'Mean Size', days_missing)
remove_depth = plants_to_remove(nums, treatments, df_depth, 'Std_Depth_Min_Leaf', days_missing)
remove_thermal = plants_to_remove(nums, treatments, df_thermal, 'Norm_avg_min_leaf_temp', days_missing)
remove_leavesNum = plants_to_remove(nums, treatments, df_leavesNum, 'Num_of_Leaves', days_missing)
remove_list = sorted(np.unique(remove_color + remove_geometric + remove_depth + remove_thermal + remove_leavesNum))

# Manually choose plants to remove
# remove = ['A14','A23','A26','A27','A33','A34','A37','A38','A42','B32','B35','B46',
#          'C25','C28','C33','C35','C41','D24','D25','D27','D35','D41','D42','D46']
Expert_remove = ['A9', 'A14', 'A23', 'A26', 'A36', 'A37', 'A38', 'A42', 'B13', 'B35',
                 'C1', 'C3', 'C7', 'C13', 'C15', 'D13', 'D22', 'D24', 'D32', 'D39', 'D40', 'D45']
remove_list = sorted(np.unique(remove_list + Expert_remove))
print("Plants to remove: ", remove_list)
print("Plants remove list size is: ", np.size(remove_list))

# Add a plant column (treatment + plant number)
df_color['Plant'] = df_color['Treatment'] + str(df_color['Num'])
df_geometric['Plant'] = df_geometric['Treatment'] + str(df_geometric['Num'])
df_depth['Plant'] = df_depth['Treatment'] + str(df_depth['Num'])
df_thermal['Plant'] = df_thermal['Treatment'] + str(df_thermal['Num'])
df_leavesNum['Plant'] = df_leavesNum['Treatment'] + str(df_leavesNum['Num'])

# Remove the plants from all indices files and then remove the 'plant' column
df_color = df_color[~(df_color['Plant'].isin(remove_list))]
df_geometric = df_geometric[~(df_geometric['Plant'].isin(remove_list))]
df_depth = df_depth[~(df_depth['Plant'].isin(remove_list))]
df_thermal = df_thermal[~(df_thermal['Plant'].isin(remove_list))]
df_leavesNum = df_leavesNum[~(df_leavesNum['Plant'].isin(remove_list))]

del(df_color['Plant'])
del(df_geometric['Plant'])
del(df_depth['Plant'])
del(df_thermal['Plant'])
del(df_leavesNum['Plant'])

# Complete the missing values in every indices file
df_color = complete_missing_values(df_color.copy(), with_index=False)
df_geometric = complete_missing_values(df_geometric.copy(), with_index=False)
df_depth = complete_missing_values(df_depth.copy(), with_index=False)
df_thermal = complete_missing_values(df_thermal.copy(), with_index=False)
df_leavesNum = complete_missing_values(df_leavesNum.copy(), with_index=False)

# Smooth all indices files
start_col = 2
df_color = smooth_df(nums, treatments, df_color.copy(), start_col, remove_list)
df_geometric = smooth_df(nums, treatments, df_geometric.copy(), start_col, remove_list)
df_depth = smooth_df(nums, treatments, df_depth.copy(), start_col, remove_list)
df_thermal = smooth_df(nums, treatments, df_thermal.copy(), start_col, remove_list)
df_leavesNum = smooth_df(nums, treatments, df_leavesNum.copy(), start_col, remove_list)

# Merge all cleaned indices files to one file and save the file
df_all = pd.merge(df_color, df_geometric, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_depth, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_thermal, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_leavesNum, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = df_all.sort_values(by=['Date', 'Treatment'])
df_all = df_all.reset_index()
df_all.to_csv(save_path + '/All_features_smoothed.csv', index=False)


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------- Option 2 ------------------------------------------------------- #
# ------------------------------ Merge all features and then clean the data together --------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

#  Load all indices files
features_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
save_path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features'
df_color = pd.read_csv(features_path + '/Features-color.csv')
df_geometric = pd.read_csv(features_path + '/Features-geometric.csv')
df_depth = pd.read_csv(features_path + '/Features-depth.csv')
df_thermal = pd.read_csv(features_path + '/Features-thermal.csv')
df_leavesNum = pd.read_csv(features_path + '/Features-num_of_leaves.csv')

# Merge all indices files to one file
df_all = pd.merge(df_color, df_geometric, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_depth, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_thermal, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = pd.merge(df_all, df_leavesNum, left_on=['Treatment', 'Num', 'Date'], right_on=['Treatment', 'Num', 'Date'])
df_all = df_all.sort_values(by=['Date', 'Treatment'])
df_all = df_all.reset_index()

# Switch treatments in indices file
df_all = switch_b_c(df_all.copy())

# Find plants to remove
nums = range(1, 49)
treatments = ['A', 'B', 'C', 'D']
days_missing = 2
remove_list = plants_to_remove(nums, treatments, df_all, 'Avg_hue', days_missing)

# Manually choose plants to remove
# remove = ['A14','A23','A26','A27','A33','A34','A37','A38','A42','B32','B35','B46',
#          'C25','C28','C33','C35','C41','D24','D25','D27','D35','D41','D42','D46']
# Expert_remove = ['A9', 'A14', 'A23', 'A26', 'A36', 'A37', 'A38', 'A42', 'B13', 'B35',
#                  'C1', 'C3', 'C7', 'C13', 'C15', 'D13', 'D22', 'D24', 'D32', 'D39', 'D40', 'D45']
Expert_remove = ['A42', 'B41', 'C41', 'D32']
remove_list = sorted(np.unique(remove_list + Expert_remove))
print("Plants to remove: ", remove_list)
print("Plants remove list size is: ", np.size(remove_list))

# Remove the plants from the indices file and then remove the 'plant' column
df_all['Plant'] = df_all['Treatment'] + df_all['Num'].astype(str)
df_all = df_all[~(df_all['Plant'].isin(remove_list))]
del(df_all['Plant'])

# Complete the missing values
df_all = complete_missing_values(df_all.copy(), with_index=False)

# Save the un-smoothed indices
df_all.to_csv(save_path + '/All_features_not_smoothed.csv', index=False)

# Smooth the data
start_col = 3
df_all = smooth_df(nums, treatments, df_all.copy(), start_col, remove_list)

# Save the smoothed indices
df_all.to_csv(save_path + '/All_features_smoothed.csv', index=False)


# -------------------------------------------- Tal Hacham Functions -------------------------------------------------- #

def find_plant_contour(img_hyper, img_directory, plot_name, path, is_save_img=True):
    # img_directory = img_name.split('\\')[0]
    # save_rgb(result_path + "/RGB.png", img_hyper, [430, 179 + 20, 108])
    frame = cv2.imread(r'D:\yotam\Phenomix\EX6_WaterStress\Images\same_size_pics\RGB\task_767_A_02.jpeg')
    ### HSV green filter - binary image
    lower_green = np.array([60 - 50, 50, 15])
    upper_green = np.array([60 + 10, 255, 255])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert to HSV
    # cv2.imwrite(plot_name + "_HSV.png", frame)  # save HSV
    mask = cv2.inRange(hsv, lower_green, upper_green)  # set pixels out of range -0, in range - 255
    # cv2.imwrite(img_directory + "/"  + plot_name + "_mask2.png", mask)   #save mask
    cv2.imwrite(img_directory + "/" + plot_name + "_mask.png", mask)  # save mask
    ### Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # find the contours
    areas = [cv2.contourArea(c) for c in contours]  # cal the area of the contours
    max_index = np.argmax(areas)  # the contour that contains max area is the plant
    contour = contours[max_index]
    cv2.drawContours(frame, contour, -1, (255, 255, 255), 1)
    if is_save_img:
        # cv2.imwrite(img_directory + "/" + plot_name + "_contour2.png", frame)   #save mask
        cv2.imwrite(path + "/" + plot_name + "_contour.png", frame)  # save mask
    return contour


def band_white_histogram(array, b, image_dir, plot_name, is_save_img=True):
    # array = array.reshape(-1,array.shape[3])
    vector_b = array[:, b]
    hist, bin_edges = np.histogram(vector_b, bins=100)
    max_index = np.argmax(hist[0:(len(hist) - 3)])
    white = bin_edges[max_index]
    # plt.figure()
    plt.cla()
    plt.hist(vector_b, bins=100)  # calculating histogram
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('white histogram plot: ' + plot_name + ' band: ' + b.__str__() + ' peak value: ' + white.__str__())
    if is_save_img:
        plt.savefig(image_dir + "/" + plot_name + "_white_histogram_plot" + '_band_' + b.__str__() + '.jpg')


def black_hist_in_band(dark_image, band, plot_name):
    # plt.figure()
    plt.cla()
    plt.hist(dark_image[:, :, band].reshape([-1]), 1000)  # calculating histogram
    plt.xlabel('Dark Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('dark histogram plot: ' + plot_name)
    plt.savefig(black_hist_dir + "/" + plot_name + "dark_hist_band_" + str(band) + '.jpg')


def black_LineChart_in_band(dark_image, band, plot_name):
    plt.cla()
    y_val = dark_image[:, :, band].reshape([-1])
    x = np.arange(len(y_val))
    # plt.figure()
    plt.bar(x, y_val, align='center', alpha=0.5)
    plt.xlabel('Width pixels')
    plt.ylabel('average value')
    plt.title('dark histogram plot: ' + plot_name)
    plt.savefig(black_hist_dir + "/" + plot_name + "dark_bar_band_" + str(band) + '.jpg')


def normalized_hist(normalized_img, plot_name):
    # plt.figure()
    plt.cla()
    plt.hist(normalized_img.reshape([-1]), 1000, [-0.5, 1])  # calculating histogram
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title('normalized histogram plot: ' + plot_name)
    # if is_save_img:
    plt.savefig(normalized_hist_dir + "/" + plot_name + "_normalized_pixel_hist" + '.jpg')


def create_hist(img, plot_name, title, dir, bins, is_save_img=True):
    # plt.figure()
    plt.cla()
    plt.hist(img.reshape([-1]), bins=bins)  #
    plt.xlabel('Pixel values')
    plt.ylabel('No. of pixels')
    plt.title(title + ': ' + plot_name)
    if is_save_img:
        plt.savefig(dir + "/" + plot_name + title + '.jpg')




