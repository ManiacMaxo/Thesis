from time import time
from typing import Tuple

import matplotlib.pyplot as plt
from numpy import ndarray
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Optimizer


def train(
    dataset: Tuple[ndarray, ndarray, ndarray, ndarray],
    model: Model,
    epochs: int,
    optimizer: Optimizer,
    validation_freq=1,
    callbacks=[],
    verbose=0,
) -> Model:
    '''Train model

    Args:
        dataset (Tuple[ndarray, ndarray, ndarray, ndarray]): dataset to train on
        model (Model): keras model
        epochs (int): number of epochs to train
        optimizer (Optimizer): keras optimizer
        validation_freq (int, optional): frequency of validation. Defaults to 1.
        callbacks (list, optional): callbacks for training. Defaults to [].
        verbose (int, optional): verbosity level for fit. Defaults to 0.

    Returns:
        Model: trained model
    '''
    start_time = time()

    X_train, y_train, X_test, y_test = dataset

    model.compile(optimizer=optimizer, loss='mse')

    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test),
                        validation_freq=validation_freq,
                        verbose=verbose)

    passed_epochs = len(history.history['loss'])
    plt.plot(range(passed_epochs), history.history['loss'], label='loss')
    plt.plot(range(validation_freq, passed_epochs + 1, validation_freq),
             history.history['val_loss'],
             label='val loss')
    plt.legend()
    plt.show()

    print(f'Training time: {(time() - start_time)/60:.2f} minutes')
    return model
