from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

# create sequences of features to query the rnn with unlabeled data
def create_feature_sequences(features, seq_length):
    # Generate sequences and corresponding labels
    sequences = []

    # pad the first few sequences with zero vectors
    empty_vectors = np.zeros((seq_length-1, features.shape[1], features.shape[2]))
    features = np.vstack((empty_vectors, features))

    for i in range(len(features)-seq_length+1):
        sequence = features[i:i + seq_length] 
        sequences.append(sequence.transpose(0, 2, 1))
        
    sequences = torch.tensor(sequences, dtype=torch.float32)

    return sequences

# create sequences of examples to learn the temporal relationship between them.
def create_data_sequences(features, labels, seq_length):
    # Generate sequences and corresponding labels
    sequences = []
    sequence_labels = []

    # pad the first few sequences with zero vectors
    empty_vectors = np.zeros((seq_length-1, features.shape[1], features.shape[2]))
    features = np.vstack((empty_vectors, features))


    for i in range(len(features)-seq_length+1):
        sequence = features[i:i + seq_length] 
        sequences.append(sequence.transpose(0, 2, 1))
        
        # Use the label of the last example in the sequence as the sequence label
        sequence_labels.append(labels[i])

    sequences = torch.tensor(sequences, dtype=torch.float32)
    sequence_labels = torch.tensor(sequence_labels, dtype=torch.long)


    return sequences, sequence_labels

# transforms data into magnitudes for linear and rotational acceleration
def transform_new_data(data):
    acc_data = data[:,:, 0:3]
    gyr_data = data[:,:, 3:6]

    total_acc = np.linalg.norm(acc_data, axis=-1)
    total_gyr = np.linalg.norm(gyr_data, axis=-1)

    x = acc_data[:, :, 0]
    y = acc_data[:, :, 1]
    z = acc_data[:, :, 2]


    jerk_x = np.diff(x, axis=1)
    jerk_y = np.diff(y, axis=1)
    jerk_z = np.diff(z, axis=1)

    jerk_x = np.hstack((np.zeros((jerk_x.shape[0], 1)), jerk_x))
    jerk_y = np.hstack((np.zeros((jerk_y.shape[0], 1)), jerk_y))
    jerk_z = np.hstack((np.zeros((jerk_z.shape[0], 1)), jerk_z))

    jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)



    return np.stack((total_acc, total_gyr, jerk_magnitude), axis=-1)

def compute_accuracy(outputs, targets):
    predicted_classes = torch.argmax(outputs, dim=1)
    accuracy = accuracy_score(targets, predicted_classes.cpu())
    return accuracy

# function for loading in all of the data
def load_full_data(train_end_index=3511, include_generated_data=True):
    given_train_data, given_train_labels, given_test_data, given_test_labels = load_real_data(train_end_index=train_end_index)

    if include_generated_data:
        gen_train_data, gen_train_labels, gen_test_data, gen_test_labels = load_generated_data(train_end_index=train_end_index)

        train_data = np.concatenate((given_train_data, gen_train_data), axis=0)
        train_labels = np.concatenate((given_train_labels, gen_train_labels), axis=0)
        test_data = np.concatenate((given_test_data, gen_test_data), axis=0)
        test_labels = np.concatenate((given_test_labels, gen_test_labels), axis=0)

        return train_data, train_labels, test_data, test_labels

    else:

        return given_train_data, given_train_labels, given_test_data, given_test_labels


def load_real_data(train_end_index=3511, sensor_names=['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']):
    # load in the given "real" data
    # this loading of generated and real data are done seperately to ensure that distrubtions between train and test sets are even.

    labels = np.loadtxt('labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt(sensor_names[0] + '_train_1.csv',
                                delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                        len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            sensor_names[sensor_index] + '_train_1.csv', delimiter=',')
        

    acc_data = data[:,:, 0:3]
    gyr_data = data[:,:, 3:6]

    total_acc = np.linalg.norm(acc_data, axis=-1)
    total_gyr = np.linalg.norm(gyr_data, axis=-1)


    total_data = np.stack((total_acc, total_gyr), axis=-1)

    given_train_data = total_data[:train_end_index+1, :, :]
    given_train_labels = labels[:train_end_index+1]
    given_test_data = total_data[train_end_index+1:, :, :]
    given_test_labels = labels[train_end_index+1:]

    return given_train_data, given_train_labels, given_test_data, given_test_labels

def load_generated_data(train_end_index=3511, sensor_names=['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']):
    # load in generated data

    gen_labels = np.loadtxt('generated_labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt('generated_' + sensor_names[0] + '_train_1.csv',
                                delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                        len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            'generated_' + sensor_names[sensor_index] + '_train_1.csv', delimiter=',')
        

    acc_data = data[:,:, 0:3]
    gyr_data = data[:,:, 3:6]

    total_acc = np.linalg.norm(acc_data, axis=-1)
    total_gyr = np.linalg.norm(gyr_data, axis=-1)


    total_data = np.stack((total_acc, total_gyr), axis=-1)

    gen_train_data = total_data[:train_end_index+1, :, :]
    gen_train_labels = gen_labels[:train_end_index+1]
    gen_test_data = total_data[train_end_index+1:, :, :]
    gen_test_labels = gen_labels[train_end_index+1:]

    return gen_train_data, gen_train_labels, gen_test_data, gen_test_labels

# slimmed down verison of the model training function without all of the diagnoistic outputs
def train_model(model, dataloader, epochs, learning_rate=0.001, criterion=nn.CrossEntropyLoss(),
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()

        for batch_sequences, batch_labels in dataloader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            outputs = model(batch_sequences)

            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f"Epoch [{epoch+1}/{epochs}]")


# train model function meant for tuning, gives diagnoistic data during training
def train_model_verbose(model, dataloader, test_sequences, test_sequence_labels,
                         epochs, learning_rate=0.001, criterion=nn.CrossEntropyLoss(),
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_acc = []
    test_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for batch_sequences, batch_labels in dataloader:
            batch_sequences, batch_labels = batch_sequences.to(device), batch_labels.to(device)
            
            outputs = model(batch_sequences)

            loss = criterion(outputs, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            running_acc += compute_accuracy(outputs, batch_labels.cpu())

        avg_acc = running_acc/len(dataloader)
        train_acc.append(avg_acc)

        model.eval() 
        with torch.no_grad():        
            test_outputs = model(test_sequences.to(device))
            avg_test_acc = compute_accuracy(test_outputs, test_sequence_labels - 1)
            test_acc.append(avg_test_acc)
            print(f"Epoch [{epoch+1}/{epochs}], Train Accuracy: {avg_acc:.4f}, Test Accuracy: {avg_test_acc:.4f}")


    plt.plot(train_acc, label='Training Acc')
    plt.plot(test_acc, label='Testing Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Training Acc vs. Epochs')
    plt.legend()
    plt.show()