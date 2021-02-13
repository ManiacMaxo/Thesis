import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import Model
from typing import Union


def AUC(model, X: np.ndarray, y: np.ndarray, tolerance=3, output=dict):
    '''AUC

    Args:
        model: tensorflow model
        X (ndarray): samples
        y (ndarray): labels
        tolerance (int, optional): tolerance for ROC. Defaults to 3.
        output ([type], optional): return type. Defaults to dict.

    Returns:
        ndarray: true, false positives and true, false negatives
    '''
    assert len(X) == len(y)
    if (len(X.shape) < 3):
        X = np.expand_dims(X, 0)
    correct = []
    predictions = model.predict(X)

    for i in range(len(X)):
        pred = (predictions[i] < tolerance).all()
        real = (y[i] < tolerance).all()
        same = (pred == real)
        if same and pred == True:
            correct.append('true positive')
        elif same and pred == False:
            correct.append('true negative')
        elif not same and pred == True:
            correct.append('false positive')
        elif not same and pred == False:
            correct.append('false negative')

    if (output == dict):
        freq = [correct.count(entry) for entry in correct]
        return dict(list(zip(correct, freq)))
    return correct


def ROC(model, X: np.ndarray, y: np.ndarray, tolerances: [], plot=True):
    histogram_x = []
    histogram_y = []

    for tol in tolerances:
        output = AUC(model, X, y, tolerance=tol)

        # all positive labels
        positives = output['true positive'] + output['false positive']
        # TPR = Σ True positive / Σ Condition positive
        true_positive_rate = output['true positive'] / positives
        # FPR = Σ False positive / Σ Condition positive
        false_positive_rate = output['false positive'] / positives

        histogram_x.append(false_positive_rate)
        histogram_y.append(true_positive_rate)

    if plot:
        plot_ROC(histogram_x, histogram_y)
    else:
        return histogram_x, histogram_y


def plot_ROC(x: [], y: []):
    plt.title('ROC curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.plot(x, y)
    plt.xlim([0, 1])
    plt.show()
