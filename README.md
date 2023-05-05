# qcnn-scr
A Low-Complexity Three Channel Quantum CNN for Speech Command Recognition

This is the official repository for the paper of the same title above.

This is for the task of speech command recognition using a quantum CNN.

The methodology is described in the paper. Here is the flowchart describing the methodology.

![methodology_flowchart_new](https://user-images.githubusercontent.com/81962282/236493614-3e9db67e-efed-4780-b1e5-b5919261b7fd.png)


You can use the Google Speech Commands V1 dataset for this task: https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html

First look at Spectrogram_Generation.ipynb as it describes the generation of spectrograms from the input dataset.
Then, go through the quanvolution.py file, as it describes the actual quanvolution (quantum convolution) of the spectrograms.
Then go through Model_Training.ipynb as it describes training the model on the quanvolved images.
Finally, you can test the model using the Test_Model.ipynb file.

You can find the Python package requirements for this project in requirements.txt.
We use a truncated version of the Google Speech Commands V1 dataset, truncated to 10 commands, which you can find here: https://drive.google.com/file/d/1Qv2FcZ2EAKEFG2N5uz52mpvEAu3LZZSq/view?usp=sharing

You can find sample 2x2 quantum convolved images on log power mel spectrograms here: https://drive.google.com/drive/folders/1hEGrWf4J6dvwDuEyLWWbTsJ1sI9r614L?usp=sharing

You can also find the model trained on the above quanvolved images named model_dep.h5
