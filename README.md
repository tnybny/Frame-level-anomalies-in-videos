# Frame-level anomaly detection in videos

Perform anomaly detection in videos using neural network architectures such as 2D convolutional auto-encoder[2] and spatial-temporal auto-encoder[3]. The focus is on finding frame-level anomalies in the UCSD Ped1 dataset[1, 5].

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
2. data.nosync/
    * (empty): space for train.tfrecords, test.tfrecords, frame-level annotation files created using src/create_tfrecords.py and src/create_<dataset>_frame_annotation.py.
3. models.nosync/
    * (empty): space for saved model using TensorFlow's saver methods.
4. results/
    * (empty): space for log files, plots and data structures that could be useful for post processing.
5. src/
    * evaluation/* : space for routines used to evaluate quality of anomaly detection (frame and pixel-level AUCs).
    * create_ped1_frame_annotation.py: creates frame annotation to guide frame-level AUC calculation which is used to guide training.
    * create_ped2_frame_annotation.py: creates frame annotation to guide frame-level AUC calculation which is used to guide training.
    * create_streetscene_frame_annotation.py: creates frame annotation to guide frame-level AUC calculation which is used to guide training.
    * conv_AE_2D.py: implements a 2D convolutional auto-encoder.
    * conv_lstm_cell.py: implements a convLSTM cell to be used in an RNN. Credit: [4].
    * create_tfrecords.py: creates train.npy and test.npy from a video anomaly detection dataset's raw data by some preprocessing.
    * data_iterator.py: tf.data pipeline feeds batches of preprocessed video clips for training and testing.
    * plots.py: implements plotting functions for results from a run.
    * spatial_temporal_autoencoder.py: implements a spatial-temporal auto-encoder which is an RNN that uses convLSTM cells in between conv and deconv of a convAE.
    * train.py: implements functions to run the network in training and testing modes by interacting with the data iterator and a model.
6. main.py: read the config file, start logging, initialize data iterator and model builder and perform training.

* Note: src/evaluation/compute_frame_roc_auc and src/evaluation/compute_pixel_roc_auc cannot be made available due to copyright.
They are not essential to this repo; details on how to implement them can be found in [1, 5].

## Instructions for usage
1. Run src/create_<dataset_name>_frame_annotation.py.
2. Set DATA_DIR and EXT in config/config.ini and run src/create_tfrecords.py.
3. Set all variables in config/config.ini and run main.py.

## Authors
1. Bharathkumar "Tiny" Ramachandra: tnybny at gmail dot com
2. Zexi "Jay" Chen

## References
1. Mahadevan, Vijay, et al. "Anomaly detection in crowded scenes." Computer Vision and Pattern Recognition (CVPR), 2010 IEEE Conference on. IEEE, 2010.
2. Hasan, Mahmudul, et al. "Learning temporal regularity in video sequences." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016.
3. Chong, Yong Shean, and Yong Haur Tay. "Abnormal event detection in videos using spatiotemporal autoencoder." International Symposium on Neural Networks. Springer, Cham, 2017.
4. https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py
5. Li, Weixin, Vijay Mahadevan, and Nuno Vasconcelos. "Anomaly detection and localization in crowded scenes." IEEE transactions on pattern analysis and machine intelligence 36.1 (2014): 18-32.
