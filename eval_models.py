import argparse
from os import listdir
from sys import argv
from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import math
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from utils import auc, load_dataset, plot_roc, train

parser = argparse.ArgumentParser(description='LSTM training')
parser.add_argument('--model', type=int, defalt=0,
                    help='model to load from model_saves/eval_models')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train for (default: 500)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learnig rate (default: 1e-3)')
parser.add_argument('--dataset', type=int, default=100,
                    help='dataset to train (default: all)')
parser.add_argument('--pth_dir', type=str, default='model_saves/evals',
                    help='where to save model checkpoints (default: model_saves/evals)')


def schedule(epoch, lr) -> float:
    if epoch >= 200 and epoch % 25 == 0:
        lr *= math.exp(-0.1)

    return lr


if __name__ == '__main__':
    args = parser.parse_args()

    scheduler = LearningRateScheduler(schedule)
    es = EarlyStopping(monitor='loss', patience=15, verbose=1)

    optimizer = Adam(lr=args.lr)
    epochs = args.epochs
    validation_freq = 5
    datasets = [args.dataset] if args.dataset !== 100 else [0, 1, 2, 3, 5, 8]

    start_time = time()

    for n in datasets:
        print(
            f'------------- Starting model {args.model} on noise {n} --------------'
        )
        lstm = load_model(f'model_saves/eval_models/model_{args.model}.h5', compile=False)
        X_train, y_train, X_test, y_test = load_dataset(f'm{n}')

        model = train(dataset=(X_train, y_train, X_test, y_test),
                      model=lstm,
                      epochs=epochs,
                      verbose=0,
                      validation_freq=validation_freq,
                      optimizer=optimizer,
                      callbacks=[scheduler, es])

        for boundary in [300, 500, 1000, 2500, 5000]:
            # plot_roc(y_test, model.predict(X_test), boundary)
            print(auc(y_test, model.predict(X_test), boundary))

        model.save(f'{args.pth_dir}/model_{args.model}-{n}.h5')

    print(f'Total time: {(time() - start_time)/60:.2f} minutes')
