import numpy as np
from tensorflow.keras import Model

def AUC(model, X: np.ndarray, y: np.ndarray, tolerance=5):
    a = []
    for i in range(len(X)): # if close to tolerance -> no transcription
        pred = not np.allclose(model.predict(np.expand_dims(X[i], 0)), 0, atol=tolerance)
        real = not np.allclose(y[i], 0, atol=tolerance)
#         correct.append('correctly predicted' if pred == real else 'incorrectly predicted')
    if pred == real and pred == True:
        correct.append('true positive')
    elif pred == real and pred == False:
        correct.append('true negative')
    elif pred != real and pred == True:
        correct.append('false positive')
    elif pred != real and pred == False:
        correct.append('false negative')

    return correct