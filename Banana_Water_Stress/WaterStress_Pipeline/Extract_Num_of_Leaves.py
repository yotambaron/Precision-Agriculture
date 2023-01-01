import os
import numpy as np
import itertools
import pandas as pd


def compute_num_of_leaves():
    # Set path and all treatments and dates
    path = r'C:\Users\yotam\Desktop\yotam\Phenomix\EX6_WaterStress\Annotations\Leaves_inside_Plant'
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
    df['Num_of_leaves'] = -1

    dir_list = os.listdir(path)
    for ind, folder in enumerate(dir_list):

        # Find the current index in the data frame
        treatment = folder[9]
        date = int(dates_dic[folder[:8]])
        plant_num = int(folder[12]) if folder[11] == 0 else int(folder[11:13])
        index = df.index[(df['Treatment'] == treatment) & (df['Num'] == plant_num) & (df['Date'] == date)]

        # Count the number of objects in the plant's annotation folder (subtract 1 for the image in the folder)
        try:
            num_of_leaves = \
                len([name for name in os.listdir(path + '/' + folder) if
                     os.path.isfile(os.path.join(path + '/' + folder, name))]) - 1
            df.at[index, 'Num_of_leaves'] = num_of_leaves
        except:
            print("No leaves annotations: ", folder)

        # Save the dataframe
        print("Finished computing plant: ", folder)
        df.to_csv(save_path + '/Features-num_of_leaves.csv', index=False)


if __name__ == "__main__":
    compute_num_of_leaves()
