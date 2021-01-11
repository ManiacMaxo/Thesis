import numpy as np
from tensorflow.keras import Model

def AUC(model, X: np.ndarray, y: np.ndarray, tolerance=3, output=dict):
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