import numpy as np
import pandas as pd
import tqdm

df = pd.read_csv(
    r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\Features-thermal.csv')
df['plant'] = df['Treatment'] + df['Num'].astype(str)
plants = np.unique(df['plant'])
df.sort_values(by='plant')

for i in range(3, len(df.columns) - 1):  # Go through all the columns in the dataframe
    slope_values = []
    for ind, plant in enumerate(plants):  # Go through every plant separately
        temp_plant = df[df['plant'] == plant]
        temp_plant = temp_plant.reset_index()
        for j in range(temp_plant.shape[0]):  # Go through the current plant's values by date
            if temp_plant.at[j, 'Date'] == 1:  # First date slope is 0
                slope_values.append(0)
            else:  # Slope is current value minus the value the day before
                slope_values.append(
                    temp_plant.at[j, temp_plant.columns[i + 1]] - temp_plant.at[j - 1, temp_plant.columns[i + 1]])
    df['Slope_' + df.columns[i]] = slope_values  # Add slope values to the dataframe

# Clean and save the dataframe
del df['plant']
df.sort_values(by=['Treatment', 'Num', 'Date'])
df.to_csv(
    r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Features\Leaves_In_Plant_New_Features\with_slopes\Features-thermal_slope.csv',
    index=False)

# ---- creating derivative features ---- #
days = 41
X = df.copy()
n_plants = int(X.shape[0] / 41)
col_names = X.columns[3:]

for plant in tqdm(range(n_plants)):
    cur_inds = np.arange(start=plant, stop=X.shape[0] + plant, step=n_plants).tolist()
    X_cur = X.copy().loc[cur_inds, col_names]
    for col in X_cur.columns:
        X.at[cur_inds, col + ' diff'] = X_cur[col].diff()
X = X.fillna(-1)
