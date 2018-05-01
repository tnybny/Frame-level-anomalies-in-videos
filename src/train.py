import logging
import numpy as np
import os
import time
from collections import defaultdict
from src.plots import plot_loss, plot_auc, plot_pfe
from src.utils import compute_eer
from sklearn.metrics import roc_auc_score, roc_curve


def train(data, model, num_iteration, result_path, model_path, print_every=200):
    logging.info("Start training the network: {}".format(time.asctime(time.localtime(time.time()))))
    frame_aucs, frame_eers, pixel_aucs, pixel_eers, losses, valid_losses = [], [], [], [], [], []
    best_auc = 0
    for i in range(num_iteration + 1):
        tr_batch = data.get_train_batch()
        loss = model.batch_train(tr_batch)
        losses.append(loss)
        if i % print_every == 0:
            logging.info("average training reconstruction loss over {0:d} iterations: {1:g}"
                         .format(print_every, np.mean(losses[-print_every:])))
            per_frame_errors, frame_auc, frame_eer, pixel_auc, pixel_eer, valid_loss = test(data, model)
            logging.info("frame level area under the roc curve at iteration {0:d}: {1:g}".format(i, frame_auc))
            logging.info("pixel level area under the roc curve at iteration {0:d}: {1:g}".format(i, pixel_auc))
            logging.info("un-regularized validation loss at iteration {0:d}: {1:g}".format(i, valid_loss))
            frame_aucs.append(frame_auc), frame_eers.append(frame_eer)
            pixel_aucs.append(pixel_auc), pixel_eers.append(pixel_eer)
            valid_losses.append(valid_loss)
            if best_auc < frame_auc:
                best_auc = frame_auc
                model.save_model(model_path)
    # store End of Training model and results
    os.makedirs(os.path.join(model_path, "EoT"))
    model.save_model(os.path.join(model_path, "EoT"))
    np.save(os.path.join(result_path, "frame_aucs.npy"), frame_aucs)
    np.save(os.path.join(result_path, "pixel_aucs.npy"), pixel_aucs)
    plot_loss(losses=losses, valid_losses=valid_losses, path=result_path)
    plot_auc(aucs=frame_aucs, path=result_path, level='Frame')
    plot_auc(aucs=pixel_aucs, path=result_path, level='Pixel')
    # store best AUC model and results
    model.restore_model(model_path)
    per_frame_errors, frame_auc, frame_eer, pixel_auc, pixel_eer, valid_loss = test(data, model)
    plot_pfe(pfe=per_frame_errors, labels=data.get_test_labels(), path=result_path)
    np.save(os.path.join(result_path, "per_frame_errors.npy"), per_frame_errors)
    return frame_auc, frame_eer, pixel_auc, pixel_eer


def test(data, model):
    data.reset_index()
    per_frame_error = [[] for _ in range(data.get_test_size())]
    re = defaultdict(lambda: defaultdict(list))
    pix_mask = data.get_pix_mask()
    while not data.check_data_exhausted():
        test_batch, frame_indices = data.get_test_batch()
        reconstruction, frame_error = model.get_reconstructions(test_batch, is_training=False)
        for i in range(frame_indices.shape[0]):
            for j in range(frame_indices.shape[1]):
                if frame_indices[i, j] != -1:
                    per_frame_error[frame_indices[i, j]].append(frame_error[i, j])
                    vid_id = frame_indices[i, j] // 200 + 1  # pix_mask keys have 1-indexing
                    if vid_id in pix_mask.keys():
                        per_pix_err = np.square(reconstruction[i, :, :, j] - test_batch[i, :, :, j])
                        re[vid_id][frame_indices[i, j] % 200].append(per_pix_err)
    per_frame_average_error = np.asarray(map(lambda x: np.mean(x), per_frame_error))
    recon_err = defaultdict(list)
    for vid_id in sorted(re.keys()):
        for frame_id in sorted(re[vid_id].keys()):
            recon_err[vid_id].append(np.mean(re[vid_id][frame_id], axis=0))
        recon_err[vid_id] = np.stack(recon_err[vid_id], axis=0)

    pix_truth, pix_pred = [], []
    for key in pix_mask.keys():
        pix_truth.append(np.ravel(np.array(pix_mask[key])))
        pix_pred.append(np.ravel(np.array(recon_err[key])))
    pix_truth, pix_pred = np.concatenate(pix_truth, axis=0) / 255, np.concatenate(pix_pred, axis=0)

    # frame-level AUC/EER
    # min-max normalize to linearly scale into [0, 1] per video
    abnorm_scores = per_video_normalize(per_frame_average_error)
    frame_auc = roc_auc_score(y_true=data.get_test_labels(), y_score=abnorm_scores)
    valid_loss = np.mean(per_frame_average_error[data.get_test_labels() == 0])
    fpr, tpr, thresholds = roc_curve(y_true=data.get_test_labels(), y_score=abnorm_scores, pos_label=1)
    frame_eer = compute_eer(far=fpr, frr=1 - tpr)

    # pixel-level AUC/EER
    pixel_auc = roc_auc_score(y_true=pix_truth, y_score=pix_pred)
    fpr, tpr, thresholds = roc_curve(y_true=pix_truth, y_score=pix_pred, pos_label=1)
    pixel_eer = compute_eer(far=fpr, frr=1 - tpr)
    return per_frame_average_error, frame_auc, frame_eer, pixel_auc, pixel_eer, valid_loss


def per_video_normalize(per_frame_error, num_frames_per_video=200):
    if per_frame_error.shape[0] % num_frames_per_video != 0:
        raise ValueError('Not all videos have same number of frames')
    num_videos = int(per_frame_error.shape[0] / num_frames_per_video)
    abnorm_scores = np.zeros((per_frame_error.shape[0], ))
    for i in range(num_videos):
        index_range = np.arange(i * num_frames_per_video, (i + 1) * num_frames_per_video)
        abnorm_scores[index_range] = (per_frame_error[index_range] - per_frame_error[index_range].min()) / \
                                     (per_frame_error[index_range].max() - per_frame_error[index_range].min())
    return abnorm_scores
