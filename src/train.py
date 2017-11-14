import logging
import numpy as np
import os
import time
from src.plots import plot_loss, plot_auc, plot_regularity
from src.utils import compute_eer
from sklearn.metrics import roc_auc_score, roc_curve


def train(data, model, num_iteration, result_path, print_every=100):
    logging.info("Start training the network: {}".format(time.asctime(time.localtime(time.time()))))
    aucs, eers, losses, valid_losses = [], [], [], []
    for i in xrange(num_iteration + 1):
        tr_batch = data.get_train_batch()
        loss = model.batch_train(tr_batch, tr_batch)
        losses.append(loss)
        if i % print_every == 0:
            logging.info("average training reconstruction loss over {0:d} iterations: {1:g}"
                         .format(print_every, np.mean(losses[-print_every:])))
            reg, auc, eer, valid_loss = test(data, model)
            logging.info("frame level area under the roc curve at iteration {0:d}: {1:g}".format(i, auc))
            logging.info("validation loss at iteration {0:d}: {1:g}".format(i, valid_loss))
            aucs.append(auc)
            eers.append(eer)
            valid_losses.append(valid_loss)
    model.save_model()
    plot_loss(losses=losses, valid_losses=valid_losses, path=result_path)
    plot_auc(aucs=aucs, path=result_path)
    plot_regularity(regularity_scores=reg, labels=data.get_test_labels(), path=result_path)
    np.save(os.path.join(result_path, "aucs.npy"), aucs)
    np.save(os.path.join(result_path, "losses.npy"), losses)
    np.save(os.path.join(result_path, "regularity_scores.npy"), reg)
    return


def test(data, model):
    data.reset_index()
    per_frame_error = [[] for _ in range(data.get_test_size())]
    while not data.check_data_exhausted():
        test_batch, frame_indices = data.get_test_batch()
        frame_error = model.get_recon_errors(test_batch, is_training=False)
        for i in xrange(frame_indices.shape[0]):
            for j in xrange(frame_indices.shape[1]):
                if frame_indices[i, j] != -1:
                    per_frame_error[frame_indices[i, j]].append(frame_error[i, j])
    per_frame_average_error = np.asarray(map(lambda x: np.mean(x), per_frame_error))
    # min-max normalize to linearly scale into [0, 1]
    abnorm_scores = per_video_abnorm_scores(per_frame_average_error)
    reg_scores = 1 - abnorm_scores
    auc = roc_auc_score(y_true=data.get_test_labels(), y_score=abnorm_scores)
    valid_loss = np.mean(per_frame_average_error[data.get_test_labels() == 0])
    fpr, tpr, thresholds = roc_curve(y_true=data.get_test_labels(), y_score=abnorm_scores, pos_label=1)
    eer = compute_eer(far=fpr, frr=1 - tpr)
    return reg_scores, auc, eer, valid_loss


def per_video_abnorm_scores(per_frame_error, num_frames_per_video=200):
    if per_frame_error.shape[0] % num_frames_per_video != 0:
        raise ValueError('Not all videos have same number of frames')
    num_videos = int(per_frame_error.shape[0] / num_frames_per_video)
    abnorm_scores = np.zeros((per_frame_error.shape[0], ))
    for i in xrange(num_videos):
        index_range = np.arange(i * num_frames_per_video, (i + 1) * num_frames_per_video)
        abnorm_scores[index_range] = (per_frame_error[index_range] - per_frame_error[index_range].min()) / \
                                     (per_frame_error[index_range].max() - per_frame_error[index_range].min())
    return abnorm_scores
