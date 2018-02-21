from __future__ import print_function, division
from src.spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from src.data_iterator import DataIterator
from src.conv_AE_2D import ConvAE2d
from src.experimental import Experiment
import ConfigParser
import logging
import os
import time
import datetime
import numpy as np
from src.train import train


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    config_path = os.path.join("config", "config.ini")
    Config.read(config_path)
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    LAMBDA = float(Config.get("Default", "LAMBDA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    TVOL = int(Config.get("Default", "TVOL"))
    TAUG = bool(int(Config.get("Default", "TAUG")))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")
    P_CLIP_PARAMS = Config.get("Default", "P_CLIP_PARAMS")
    METHOD = Config.get("Default", "METHOD")

    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    result_path = os.path.join("results", "archive", METHOD, dt)
    os.makedirs(result_path)
    logging.basicConfig(filename=os.path.join(result_path, "info.log"), level=logging.INFO)
    model_path = os.path.join("models", METHOD, dt)
    os.makedirs(model_path)

    d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE, tvol=TVOL, taug=TAUG)
    if METHOD == 'STAE':
        net = SpatialTemporalAutoencoder(tvol=TVOL, alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)
    elif METHOD == 'CONVAE2D':
        net = ConvAE2d(tvol=TVOL, alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)
    elif METHOD == 'EXP':
        net = Experiment(tvol=TVOL, alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)
    else:
        raise ValueError('Incorrect method specification')

    area_under_roc, equal_error_rate = train(data=d, model=net, num_iteration=NUM_ITER, result_path=result_path,
                                             model_path=model_path)
    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))
    logging.info("Equal error rate corresponding to best auc: {0:g}".format(equal_error_rate))
