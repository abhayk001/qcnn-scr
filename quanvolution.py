import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as qmlnp
from pennylane.templates import RandomLayers

# Load the spectrogram dataset you want to train on
data = tf.keras.utils.image_dataset_from_directory('/content/speech_commands_10_mel', image_size=(32,32))

data_iterator = data.as_numpy_iterator()

data = data.map(lambda x,y: (x/255, y))

train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data) - train_size - val_size)


# Split the dataset into training, validation and testing splits
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

# Uncomment these lines for 3x3 quanvolution

# dev = qml.device("default.qubit", wires=9)
# n_layers = 1
# # Random circuit parameters
# # These parameters can be random, but they need to be consistent for when you do testing with the trained model
# # Therefore, use any parameters, but keep them consistent, because these are akin to the convolution kernel
# # rand_params = qmlnp.random.uniform(high=2 * qmlnp.pi, size=(n_layers, 9))

# # Sample parameters, you can change these, but make the same change in the other parts of the code for testing the model
# rand_params = [[4.67569424, 1.72307178, 2.95711189, 6.189338, 4.62112322, 2.51773206, 0.05994887, 0.43047562, 0.82051428]]


# @qml.qnode(dev)
# def circuit(phi):
#     # Encoding of 9 classical input values
#     for j in range(9):
#         qml.RY(qmlnp.pi * phi[j], wires=j)

#     # Random quantum circuit
#     RandomLayers(rand_params, wires=list(range(9)))

#     # Measurement producing 9 classical output values
#     return [qml.expval(qml.PauliZ(j)) for j in range(9)]

# def quanv(image):
#     """Convolves the input image with many applications of the same quantum circuit."""
#     out = qmlnp.zeros((11, 11, 27))

#     # Loop over the coordinates of the top-left pixel of 3X3 squares. The stride used is 3
#     for j in range(0, 32, 3):
#         for k in range(0, 32, 3):

#             # Prepare the 3x3 region to send to the quantum circuit. Add zero padding when necessary.
#             # There are three lists for the three red, green and blue channels of the image.

#             q_results_0_list = []
#             q_results_1_list = []
#             q_results_2_list = []
#             for x in range(3):
#                 for y in range(3):
#                     try:
#                         q_results_0_list.append(image[j + x, k + y, 0])
#                     except:
#                         q_results_0_list.append(0)

#                     try:
#                         q_results_1_list.append(image[j + x, k + y, 1])
#                     except:
#                         q_results_1_list.append(0)

#                     try:
#                         q_results_2_list.append(image[j + x, k + y, 2])
#                     except:
#                         q_results_2_list.append(0)
                    
#             # Process a squared 3X3 region of the image with a quantum circuit. Three results for three channels (RGB)
#             q_results_0 = circuit(q_results_0_list)
#             q_results_1 = circuit(q_results_1_list)
#             q_results_2 = circuit(q_results_2_list)
            
            
#             # Assign expectation values to different channels of the output pixel (j/3, k/3)
#             for c in range(0, 9):
#                 out[j//3, k//3, c] = q_results_0[c]
#             for c in range(9, 18):
#                 out[j//3, k//3, c] = q_results_1[c-9]
#             for c in range(18, 27):
#                 out[j//3, k//3, c] = q_results_2[c-18]
#     return out



# Uncomment these for 2x2 quanvolution

# 4 qubit device for 2x2 quanvolution
dev = qml.device("default.qubit", wires=4)
n_layers = 1
# Random circuit parameters
# These parameters can be random, but they need to be consistent for when you do testing with the trained model
# Therefore, use any parameters, but keep them consistent, because these are akin to the convolution kernel
# rand_params = qmlnp.random.uniform(high=2 * qmlnp.pi, size=(n_layers, 4))

# Sample parameters, you can change these, but make the same change in the other parts of the code for testing the model
rand_params = [[2.97233695, 1.93705856, 3.75475994, 3.3502165]]

@qml.qnode(dev)
def circuit(phi):
    # Encoding of 4 classical input values
    for j in range(4):
        qml.RY(qmlnp.pi * phi[j], wires=j)

    # Random quantum circuit
    RandomLayers(rand_params, wires=list(range(4)))

    # Measurement producing 4 classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    """Convolves the input image with many applications of the same quantum circuit."""
    out = qmlnp.zeros((16, 16, 12))

    # Loop over the coordinates of the top-left pixel of 2X2 squares
    for j in range(0, 32, 2):
        for k in range(0, 32, 2):
            # Process a squared 2x2 region of the image with a quantum circuit
            q_results_0 = circuit(
                [
                    image[j, k, 0],
                    image[j, k + 1, 0],
                    image[j + 1, k, 0],
                    image[j + 1, k + 1, 0]
                ]
            )
            
            q_results_1 = circuit(
                [
                    image[j, k, 1],
                    image[j, k + 1, 1],
                    image[j + 1, k, 1],
                    image[j + 1, k + 1, 1]
                ]
            )
            
            q_results_2 = circuit(
                [
                    image[j, k, 2],
                    image[j, k + 1, 2],
                    image[j + 1, k, 2],
                    image[j + 1, k + 1, 2]
                ]
            )
            
            # Assign expectation values to different channels of the output pixel (j/2, k/2)
            for c in range(0, 4):
                out[j//2, k//2, c] = q_results_0[c]
            for c in range(4, 8):
                out[j//2, k//2, c] = q_results_1[c-4]
            for c in range(8, 12):
                out[j//2, k//2, c] = q_results_2[c-8]
    return out




q_train_images = []
train_labels = []
for batch_idx, batch in enumerate(train.as_numpy_iterator()):
  X, y = batch
  train_labels.append(y)
  print("Train Batch " + str(batch_idx))
  for idx, x in enumerate(X):
    print(str(idx) + "/32")
    q_train_images.append(quanv(x))
q_train_images = np.asarray(q_train_images)
train_labels = np.asarray(train_labels)

q_test_images = []
test_labels = []
for batch_idx, batch in enumerate(test.as_numpy_iterator()):
  X, y = batch
  test_labels.append(y)
  if len(y) != 32:
    continue
  print("Test Batch " + str(batch_idx))
  for idx, x in enumerate(X):
    print(str(idx) + "/32")
    q_test_images.append(quanv(x))
q_test_images = np.asarray(q_test_images)
test_labels = np.asarray(test_labels)

q_val_images = []
val_labels = []
for batch_idx, batch in enumerate(val.as_numpy_iterator()):
  X, y = batch
  val_labels.append(y)
  print("Val Batch " + str(batch_idx))
  for idx, x in enumerate(X):
    print(str(idx) + "/32")
    q_val_images.append(quanv(x))
q_val_images = np.asarray(q_val_images)
val_labels = np.asarray(val_labels)


# Save the results of the quanvolution at any place you want
# This saves both the quanvolution results and the corresponding commands/labels
# This will be used to train a classification model
np.save('/content/q_train_images_mel.npy', q_train_images)
np.save('/content/q_test_images_mel.npy', q_test_images)
np.save('/content/q_val_images_mel.npy', q_val_images)
np.save('/content/train_labels_mel.npy', train_labels)
np.save('/content/test_labels_mel.npy', test_labels)
np.save('/content/val_labels_mel.npy', val_labels)