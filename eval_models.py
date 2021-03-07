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

from utils import load_dataset, train


def schedule(epoch, lr) -> float:
    if epoch >= 200 and epoch % 25 == 0:
        lr *= math.exp(-0.1)

    return lr


def plot_roc(y_true, y_score, boundary):
    y = np.where(y > boundary, 0, 1)
    pred = np.where(pred > boundary, 0, 1)

    fpr, tpr, thresholds = roc_curve(y, pred)

    plt.title(f'ROC curve with boundary {boundary}')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(tpr, fpr)
    plt.show()
    print(f'Thresholds: {thresholds}')


def auc(y_true, y_score, boundary):
    y = np.where(y_true > boundary, 0, 1)
    pred = np.where(y_score > boundary, 0, 1)

    return roc_auc_score(y, pred)


scheduler = LearningRateScheduler(schedule)
es = EarlyStopping(monitor='loss', patience=10, verbose=1)
optimizer = Adam(lr=1e-3)
epochs = 1500
validation_freq = 5

start_time = time()
for fname in listdir('model_saves/eval_models'):

    lstm = load_model(f'model_saves/eval_models/{fname}')
    lstm.summary()

    for n in [0, 1, 2, 3, 5, 8]:
        print(f'-------------- Starting model {i} on noise {n} --------------')
        X_train, y_train, X_test, y_test = load_dataset(f'm{n}')

        lstm.set_weights(weights[i])  # reset weights
        model = train((X_train, y_train, X_test, y_test), lstm, epochs, 0,
                      validation_freq, [scheduler, es])

        for boundary in [300, 500, 1000, 2500, 5000]:
            # plot_roc(y_test, model.predict(X_test), boundary)
            print(auc(y_test, model.predict(X_test), boundary))

        model.save(f'model_saves/evals/{fname.replace('.h5', '')}-{n}.h5')
        print(f'Elapsed time: {time() - start_time}')

