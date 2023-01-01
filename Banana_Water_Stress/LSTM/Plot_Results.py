import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Load data
path = r'C:\Users\User\PycharmProjects\Deep_Learning_Course'
lstm_results = pd.read_csv(path + '/Final_LSTM_CV_Results.csv')
ann_results = pd.read_csv(path + '/Final_ANN_Results.csv')

ann_results = ann_results[ann_results['Time_step'] >= np.min(lstm_results['Day'])]
days = ann_results['Time_step']

plt.figure(figsize=(10, 7))
plt.plot(days, ann_results['Accuracy'] * 100, linestyle='solid')
plt.plot(days, lstm_results['Accuracy_Validation'], linestyle='solid')
plt.legend(['ANN', 'LSTM'], loc='upper right', fontsize=12)
plt.title("ANN & LSTM Results By Days", fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.grid()
plt.show()
plt.savefig(path + '/ANN_LSTM_Graph')






