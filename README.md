# Frame-level anomaly detection in videos

Perform anomaly detection in videos using neural network architectures such as 2D convolutional auto-encoder[2] and spatial-temporal auto-encoder[3]. The focus is on finding frame-level anomalies in the UCSD Ped1 dataset[1].

## Prerequisites
1. Python 2.7+
    * PIL
    * glob
    * cv2
    * numpy
    * matplotlib
    * sklearn
2. CUDA Toolkit 8+
3. TensorFlow 1.3+

## List of files and their functions
1. config/
    * config.ini: contains settings for the run such as which network to use, learning rate, batch size and etcetera.
2. data/
    * (empty): space for labels.npy, train.npy and test.npy created using src/create_dataset.py and src/create_labels.py.
3. models/
    * (empty): space for saved model using TensorFlow's saver methods.
4. results/
    * (empty): space for log files, plots and data structures that could be useful for post processing.
5. src/
    * conv_AE_2D.py: implements a 2D convolutional auto-encoder.
    * conv_lstm_cell.py: implements a convLSTM cell to be used in an RNN. Credit: [4].
    * create_dataset.py: creates train.npy and test.npy from UCSD Ped1 raw data by some preprocessing.
    * create_labels.py: creates the labels for test data of UCSD Ped1 dataset.
    * data_iterator.py: feeds batches of video clips for training and testing.
    * plots.py: implements plotting functions for results from a run.
    * spatial_temporal_autoencoder.py: implements a spatial-temporal auto-encoder which is an RNN that uses convLSTM cells in between conv and deconv of a convAE.
    * train.py: implements functions to run the network in training and testing modes by interacting with the data iterator and a model.
6. main.py: read the config file, start logging, initialize data iterator and model builder and perform training.

## Authors
1. Bharathkumar "Tiny" Ramachandra: tnybny at gmail dot com
2. Zexi "Jay" Chen

## References
1. Mahadevan, Vijay, et al. "Anomaly detection in crowded scenes." Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010.
2. Hasan, Mahmudul, et al. "Learning temporal regularity in video sequences." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
3. Chong, Yong Shean, and Yong Haur Tay. "Abnormal event detection in videos using spatiotemporal autoencoder." International Symposium on Neural Networks. Springer, Cham, 2017.
4. https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py
