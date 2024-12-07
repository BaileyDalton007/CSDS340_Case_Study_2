# -*- coding: utf-8 -*-
"""
Demo of a model using a CNN for feature extraction, then a RNN to learn temporal patterns within the data

@author: Bailey Dalton, Franklin Wang
"""

import numpy as np
from CNN_RNN_Model import CNN_RNN_Model
from helper_funcs import *

MODEL_PATH = 'combined_model.pth'
SEQUENCE_LENGTH = 5

def predict_test(train_data, train_labels, test_data):
    # load the model from a pretrained weight file
    model = CNN_RNN_Model(num_classes=4, input_channels=3, feature_length=60)
    state_dict = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()

    # transforms 6 channels into 2, being magnitude of linear and rotational acceleration
    data = transform_new_data(test_data)

    test_sequences = create_feature_sequences(data, seq_length=SEQUENCE_LENGTH)

    with torch.no_grad():
        test_outputs = model(test_sequences)

        predicted_classes = torch.argmax(test_outputs, dim=1)

        # the +1 converts the 0 indexed prediction back to class labels (1-4)
        return predicted_classes+1