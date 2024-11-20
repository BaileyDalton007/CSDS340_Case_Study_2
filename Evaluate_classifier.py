# -*- coding: utf-8 -*-
"""
Script used for pre-submission evaluation of classifier accuracy on training
set 2

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from Classify_activity import predict_test

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'
test_suffix = '_train_2.csv'

def load_sensor_data(sensor_names, suffix):
    data_slice_0 = np.loadtxt(sensor_names[0] + suffix, delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + suffix, delimiter=',')
    
    return data
    
# Load labels and sensor data into 3-D array
train_labels = np.loadtxt('labels' + train_suffix, dtype='int')
train_data = load_sensor_data(sensor_names, train_suffix)
test_labels = np.loadtxt('labels' + test_suffix, dtype='int')
test_data = load_sensor_data(sensor_names, test_suffix)

# Predict activities on test data
test_outputs = predict_test(train_data, train_labels, test_data)

# Compute micro and macro-averaged F1 scores
micro_f1 = f1_score(test_labels, test_outputs, average='micro')
macro_f1 = f1_score(test_labels, test_outputs, average='macro')
print(f'Micro-averaged F1 score: {micro_f1}')
print(f'Macro-averaged F1 score: {macro_f1}')

# Examine outputs compared to labels
n_test = test_labels.size
plt.subplot(2, 1, 1)
plt.plot(np.arange(n_test), test_labels, 'b.')
plt.xlabel('Time window')
plt.ylabel('Target')
plt.subplot(2, 1, 2)
plt.plot(np.arange(n_test), test_outputs, 'r.')
plt.xlabel('Time window')
plt.ylabel('Output (predicted target)')
plt.show()
