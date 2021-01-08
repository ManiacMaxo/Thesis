import numpy as np
from tensorflow.keras import Model

def AUC(model, X: np.ndarray, y: np.ndarray, tolerance=5):
    correct = []
    for i in range(len(X)): # if close to 5 -> no transcription
        pred = not np.allclose(model.predict(np.expand_dims(X[i], 0)), 0, atol=tolerance)
        real = not np.allclose(y[i], 0, atol=tolerance)
#         correct.append('correctly predicted' if pred == real else 'incorrectly predicted')
        correct.append(pred == real)

    return correct