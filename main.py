from __future__ import print_function, division
from src.spatial_temporal_autoencoder import SpatialTemporalAutoencoder
from src.data_iterator import DataIterator, DataIteratorStae
from src.conv_AE_2D import ConvAE2d
import ConfigParser
import logging
import os
import time
import datetime
from src.train import train


if __name__ == "__main__":
    Config = ConfigParser.ConfigParser()
    config_path = os.path.join("config", "config.ini")
    Config.read(config_path)
    NUM_ITER = int(Config.get("Default", "NUM_ITER"))
    ALPHA = float(Config.get("Default", "ALPHA"))
    LAMBDA = float(Config.get("Default", "LAMBDA"))
    GAMMA = float(Config.get("Default", "GAMMA"))
    BATCH_SIZE = int(Config.get("Default", "BATCH_SIZE"))
    TVOL = int(Config.get("Default", "TVOL"))
    TAUG = bool(int(Config.get("Default", "TAUG")))
    P_TRAIN = Config.get("Default", "P_TRAIN")
    P_TEST = Config.get("Default", "P_TEST")
    P_LABELS = Config.get("Default", "P_LABELS")
    METHOD = Config.get("Default", "METHOD")

    ts = time.time()
    dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    result_path = os.path.join("results", "archive", METHOD, dt)
    os.makedirs(result_path)
    logging.basicConfig(filename=os.path.join(result_path, "info.log"), level=logging.INFO)
    model_path = os.path.join("models", METHOD, dt)
    os.makedirs(model_path)

    if METHOD == 'STAE':
        net = SpatialTemporalAutoencoder(tvol=TVOL, alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)
        d = DataIteratorStae(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE, tvol=TVOL, taug=TAUG)
    elif METHOD == 'CONVAE2D':
        net = ConvAE2d(tvol=TVOL, alpha=ALPHA, batch_size=BATCH_SIZE, lambd=LAMBDA)
        d = DataIterator(P_TRAIN, P_TEST, P_LABELS, batch_size=BATCH_SIZE, tvol=TVOL, taug=TAUG)
    else:
        raise ValueError('Incorrect method specification')

    area_under_roc, equal_error_rate = train(data=d, model=net, num_iteration=NUM_ITER, result_path=result_path,
                                             model_path=model_path)
    logging.info("Best area under the roc curve: {0:g}".format(area_under_roc))
    logging.info("Equal error rate corresponding to best auc: {0:g}".format(equal_error_rate))
