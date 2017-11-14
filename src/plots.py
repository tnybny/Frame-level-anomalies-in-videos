import os
import numpy as np
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
    plt.plot(range(0, len(losses), valid_loss_every), valid_losses, 'r', linewidth=0.5, label="validation loss")
    plt.xlabel("Iteration")
    plt.ylabel("Total loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, "Loss.png"))


def plot_auc(aucs, path):
    """
    plot area under the curve vs. (iteration number / auc_every)
    """
    plt.figure()
    plt.plot(range(1, len(aucs) + 1), aucs)
    plt.xlabel("Training progress (# iter / constant)")
    plt.ylabel("Area under the roc curve")
    plt.savefig(os.path.join(path, "AUC.png"))


def plot_regularity(regularity_scores, labels, path):
    """
    plot regularity score vs. frame number and shade anomalous background using ground truth labels
    for each video in the test set
    """
    num_test_vids = 36
    frames_per_video = 200
    for vid_id in range(num_test_vids):
        plt.figure()
        start = vid_id * frames_per_video
        end = start + 200
        y_ax = regularity_scores[np.arange(start, end)]
        plt.plot(np.arange(1, frames_per_video + 1), y_ax, linewidth=0.5)
        plt.xlabel("Frame number")
        plt.ylabel("Regularity score")
        for i in xrange(start, end):
            if labels[i] == 1:
                plt.axvspan(i - start, i + 1 - start, facecolor='salmon', alpha=0.5)
        plt.savefig(os.path.join(path, "Regularity_vid{0:d}.png".format(vid_id + 1)))
