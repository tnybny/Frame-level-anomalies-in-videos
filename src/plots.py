import os
import numpy as np
import re
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(losses, valid_losses, path):
    """
    plot training loss vs. iteration number
    """
    plt.figure()
    plt.plot(range(len(losses)), losses, 'b', alpha=0.6, linewidth=0.5, label="training loss")
    valid_loss_every = (len(losses) - 1) / (len(valid_losses) - 1)
    plt.plot(range(0, len(losses), valid_loss_every), valid_losses, 'r', linewidth=0.5,
             label="un-regularized validation loss")
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, "Loss.png"))
    plt.close()


def plot_auc(aucs, path, level):
    """
    plot area under the curve vs. (iteration number / auc_every)
    """
    plt.figure()
    plt.plot(range(1, len(aucs) + 1), aucs)
    plt.xlabel("Training progress (# iter / constant)")
    plt.ylabel("Area under the roc curve")
    plt.savefig(os.path.join(path, level + "_AUC.png"))
    plt.close()


def plot_pfe(pfe, labels, test_dir, ext, result_path):
    """
    plot per frame error vs. frame number and shade anomalous background using ground truth labels
    for each video in the test set
    """
    test_vids = sorted([os.path.join(test_dir, d) for d in os.listdir(test_dir)
                        if re.match(r'Test[0-9][0-9][0-9]$', d)])
    start, end = 0, 0
    for vid_id in range(len(test_vids)):
        plt.figure()
        frames_in_vid = len(glob(os.path.join(test_vids[vid_id], '*.' + ext)))
        start = end
        end = start + frames_in_vid
        y_ax = pfe[np.arange(start, end)]
        plt.plot(np.arange(1, frames_in_vid + 1), y_ax, linewidth=0.5)
        plt.xlabel("Frame number")
        plt.ylabel("Reconstruction error")
        for i in range(start, end):
            if labels[i] == 1:
                plt.axvspan(i - start, i + 1 - start, facecolor='salmon', alpha=0.5)
        plt.savefig(os.path.join(result_path, "PFE_vid{0:d}.png".format(vid_id + 1)))
        plt.close()
