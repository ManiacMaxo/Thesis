from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, boundaries: list = None):
    plt.title("ROC curve")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")

    if boundaries != None:
        for boundary in boundaries:
            y = np.where(y_true > boundary, 0, 1)
            pred = np.where(y_score > boundary, 0, 1)

            fpr, tpr, thresholds = roc_curve(y, pred)
            plt.plot(fpr, tpr, label=f"bnd {boundary}")
        plt.legend()
    else:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr)
    plt.show()


def auc(y_true, y_score, boundary: int = None):
    if boundary != None:
        y = np.where(y_true > boundary, 0, 1)
        pred = np.where(y_score > boundary, 0, 1)
    else:
        y = y_true
        pred = y_score

    return roc_auc_score(y, pred)
